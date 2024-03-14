import torch
import math
import sys
from torchinfo import summary
from tqdm import tqdm

from .base_solver import BaseSolver
from .optimizer import adjust_learning_rate
from .formatter import Formatter
from .utils import averager

from benchmark import (
    batch_evaluate_concurrent, 
    batch_evaluate_scenario,
    probe_metrics,
)

class WTSSolver(BaseSolver):
  
    def __init__(self, cfg,
                 experiment_name,
                 signature,
                 local_dir,
                 model, 
                 train_loader=None,
                 val_loader=None,
                 optim=None,
                 is_eval=False,
                 hparams=None,
                 device: torch.device = torch.device("cuda")):
      
        super().__init__(cfg, signature, experiment_name, local_dir)
                
        self.device = device
        self.model = model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        
        if self.model.load_ckpt and not is_eval:
          checkpoint = torch.load(self.model.pretrained_ckpt, map_location="cpu")
          self.model.load_pretrained(checkpoint["model"])
          if "optimizer" in checkpoint:
              assert self.optim is not None
              self.optim.load_state_dict(checkpoint["optimizer"])
          print("\nPretrained Vid2Seq checkpoint has beed loaded!")
          
        summary(self.model)

        self.register_stateful('model', 'optim')
        
        if cfg.LOG_TO_WANDB and not is_eval:
            self.init_wandb(
                log_folder=local_dir,
                project="AIC Track 2",
                signature=f"{experiment_name}_{signature}",
            )
        self.hparams = hparams
        self.max_epoch = self.train_cfg.MAX_EPOCH
        self.log_updates = self.train_cfg.LOG_UPDATES
        self.checkpoint_metrics = self.train_cfg.CHECKPOINT_METRICS
        
        self.val_interval = self.val_cfg.VAL_INTERVAL
        assert self.save_interval % self.val_interval == 0
        
        self.val_metrics_list = list(probe_metrics().keys())
        self.val_metrics_list.append('loss')
        self.val_samples_history = []

    def run(self):
        self.logger.info('Log dir: %s', self.folder)
        self.restore()
        if self.hparams is not None:
            self.log_hyperparams(self.hparams)
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.run_stage("train", self.do_train)
            if epoch % self.val_interval == 0:
                self.run_stage("valid", self.do_valid)
            self.commit()

    def get_formatter(self, stage_name: str):
        assert self.val_loader is not None
        if stage_name == "train":
            return Formatter({
                'loss': '.5f',
                'lr': 'e',
            })
        elif stage_name == "valid":
            base_format = {
                _metric: '.5f' for _metric in self.val_metrics_list
            }
            ds_names = [
                f"{name}/{branch}" for name, branch in self.val_loader.dataset.all_sub_ds
            ]
            return Formatter({
                f"{ds_name}/{k}": v for ds_name in ds_names for k, v in base_format.items()
            })
        raise NotImplementedError()

    def do_train(self):
        assert self.train_loader is not None and \
            self.optim is not None
        self.logger.info('-' * 80)
        self.logger.info(f'Starting {self.current_stage} stage...')
        loader = self.train_loader
        num_samples = len(loader)
        lp = self.log_progress(
            self.current_stage, loader, total=num_samples, updates=self.log_updates, use_tqdm = True
        )
        average = averager()
        self.model.train()
        num_training_steps = int(num_samples * self.max_epoch)

        for idx, batch in tqdm(enumerate(lp), leave=False, total=num_samples, 
                               desc=f"Epoch {self.epoch}/{self.max_epoch}"):
            feat = batch["feat"].to(self.device)
            vehicle_tokens = batch["vehicle_tokens"].to(self.device)
            pedestrian_tokens = batch["pedestrian_tokens"].to(self.device)
            
            vehicle_loss = self.model(feat, vehicle_tokens, "vehicle")
            pedestrian_loss = self.model(feat, pedestrian_tokens, "pedestrian")
            if not math.isfinite(vehicle_loss.item()) or \
                not math.isfinite(pedestrian_loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
                
            loss = vehicle_loss + pedestrian_loss
            
            self.optim.zero_grad()
            loss.backward()
            if self.train_cfg.CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_( # type: ignore
                    self.model.parameters(), self.train_cfg.CLIP_MAX_NORM
                )
            self.optim.step()
            
            self.model.normalize_time_embeddings()
            adjust_learning_rate(
                self.optim,
                curr_step=(self.epoch - 1) * num_samples + idx,
                num_training_steps=num_training_steps,
                cfg=self.optim_cfg,
            )
            
            metrics = average({'loss': loss, 'lr': self.optim.param_groups[0]["lr"]})
            lp.update(**metrics)

        return metrics
    
    @torch.no_grad()
    def generate_samples(self, feat, tgt_type, max_output_tokens):
        return self.model.generate(
            feats=feat,
            tgt_type=tgt_type,
            use_nucleus_sampling=self.val_cfg.NUM_BEAMS == 0,
            num_beams=self.val_cfg.NUM_BEAMS,
            max_length=max_output_tokens,
            min_length=1,
            top_p=self.val_cfg.TOP_P if self.val_cfg.NUM_BEAMS == 0 else 1.0,
            repetition_penalty=self.val_cfg.REPETITION_PENALTY,
            length_penalty=self.val_cfg.LENGTH_PENALTY,
            num_captions=1,
            temperature=self.val_cfg.TEMPERATURE,
        )
    
    @torch.no_grad()
    def do_valid(self):
        assert self.val_loader is not None
        self.logger.info('-' * 80)
        self.logger.info(f'Starting {self.current_stage} stage...')
        loader = self.val_loader
        all_metrics = {}
        loader.dataset.reset_counter()
        
        while loader.dataset.next_dataset():
            num_samples = len(loader)
            lp = self.log_progress(
                self.current_stage, loader, total=num_samples, updates=self.log_updates, use_tqdm = True
            )
            average = averager()
            self.model.eval()

            for idx, batch in tqdm(enumerate(lp), leave=False, total=num_samples, 
                                   desc=f"Epoch {self.epoch}/{self.max_epoch}"):
                feat = batch["feat"].to(self.device)
                vehicle_tokens = batch["vehicle_tokens"].to(self.device)
                pedestrian_tokens = batch["pedestrian_tokens"].to(self.device)
                vehicle_text = batch["vehicle_text"]
                pedestrian_text = batch["pedestrian_text"]
                
                pred_vehicle_text = self.generate_samples(
                    feat, "vehicle", loader.dataset.max_output_tokens
                )
                pred_pedestrian_text = self.generate_samples(
                    feat, "pedestrian", loader.dataset.max_output_tokens
                )
                vehicle_loss = self.model(feat, vehicle_tokens, "vehicle")
                pedestrian_loss = self.model(feat, pedestrian_tokens, "pedestrian")
                loss = vehicle_loss + pedestrian_loss
                
                vehicle_metrics = batch_evaluate_scenario(pred_vehicle_text, vehicle_text)
                vehicle_total = vehicle_metrics["TOTAL"]
                vehicle_metrics = {
                    f"{loader.dataset.curr_ds_name}/vehicle_out/{k}": v 
                    for k, v in vehicle_metrics.items()
                }
                pedestrian_metrics = batch_evaluate_scenario(pred_pedestrian_text, pedestrian_text)
                pedestrian_total = pedestrian_metrics["TOTAL"]
                pedestrian_metrics = {
                    f"{loader.dataset.curr_ds_name}/pedestrian_out/{k}": v 
                    for k, v in pedestrian_metrics.items()
                }
                fin_total = (vehicle_total + pedestrian_total) / 2
                
                metrics = average(
                    {f"{loader.dataset.curr_ds_name}/loss": loss, 
                     f"{loader.dataset.curr_ds_name}/TOTAL": fin_total, 
                     **vehicle_metrics, **pedestrian_metrics}
                )
                all_metrics = all_metrics | metrics
                lp.update(**metrics)
                
                if idx == 0:
                    self.val_samples_history.append(
                        [str(self.epoch), pred_vehicle_text[0], vehicle_text[0],
                         pred_pedestrian_text[0], pedestrian_text[0]]
                    )
                    self.log_text(
                        self.current_stage, 
                        ["epoch",
                         "vehicle_pred",
                         "vehicle_gt",
                         "pedestrian_pred",
                         "pedestrian_gt"],
                        self.val_samples_history,
                    )

        return all_metrics
    
    # @torch.no_grad()
    # def generate_to_len(self, feat, max_output_tokens, tgt_type, scenarios, num_phases, max_trials=5):
    #     texts = self.generate_samples(feat, tgt_type, max_output_tokens)
    #     parsed_texts = batch_parse(texts)
    #     assert len(scenarios) == len(parsed_texts)
        
    #     invalid_indices = []
    #     results = {}
    #     for idx, scenario in enumerate(scenarios):
    #         scenario_texts = parsed_texts[idx]
    #         if len(scenario_texts) < num_phases[idx]:
    #             if max_trials == 0:
    #                 print(f"MAX TRIALS REACHED! Duplicating last text for {scenario}'s {tgt_type}...")
    #                 dup = scenario_texts[-1]
    #                 for _ in range(num_phases[idx] - len(scenario_texts)):
    #                     scenario_texts.append(dup)
    #             else:
    #                 invalid_indices.append(idx)
    #                 continue
    #         results[scenario] = scenario_texts
    #     if len(invalid_indices) == 0:
    #         return results
    #     return self.generate_to_len(
    #         torch.index_select(feat, 0, torch.tensor(invalid_indices)),
    #         max_output_tokens,
    #         tgt_type,
    #         [scenarios[i] for i in invalid_indices],
    #         [num_phases[i] for i in invalid_indices],
    #         max_trials - 1
    #     ) | results
        
    # @torch.no_grad()
    # def generate_to_len_once(self, feat, max_output_tokens, tgt_type, scenarios, num_phases):
    #     texts = self.generate_samples(feat, tgt_type, max_output_tokens)
    #     parsed_texts = batch_parse(texts)
    #     assert len(scenarios) == len(parsed_texts)
        
    #     results = {}
    #     for idx, scenario in enumerate(scenarios):
    #         scenario_texts = parsed_texts[idx]
    #         if len(scenario_texts) < num_phases[idx]:
    #             print(f"MAX TRIALS REACHED! Duplicating last text for {scenario}'s {tgt_type}...")
    #             dup = scenario_texts[-1]
    #             for _ in range(num_phases[idx] - len(scenario_texts)):
    #                 scenario_texts.append(dup)
    #         results[scenario] = scenario_texts
    #     return results
    
    # @torch.no_grad()
    # def do_test(self, loader):
    #     assert self.load_path is not None
    #     self.restore()
        
    #     self.logger.info('-' * 80)
    #     self.logger.info(f'Start evaluating {self.model.__class__.__name__} at epoch {str(self.load_from_epoch)}...')
    #     loader.dataset.reset_counter()
        
    #     return_dict = {}
    #     while loader.dataset.next_dataset():
    #         num_samples = len(loader)
    #         lp = self.log_progress(
    #             self.current_stage, loader, total=num_samples, updates=self.log_updates, use_tqdm = True
    #         )
    #         self.model.eval()

    #         for _, batch in tqdm(enumerate(lp), leave=False, total=num_samples, 
    #                                desc=loader.dataset.curr_ds_name):
    #             feat = batch["feat"].to(self.device)
    #             scenarios = batch["scenario"]
    #             label_order = batch["label_order"]
    #             num_phases = [len(label) for label in label_order]
                
    #             vehicle_dict = self.generate_to_len_once(
    #                 feat, loader.dataset.max_output_tokens, "vehicle", scenarios, num_phases
    #             )
    #             pedestrian_dict = self.generate_to_len_once(
    #                 feat, loader.dataset.max_output_tokens, "pedestrian", scenarios, num_phases
    #             )
    #             for scenario_idx, scenario in enumerate(scenarios):
    #                 assert scenario not in return_dict
    #                 vehicle_txts = vehicle_dict[scenario]
    #                 pedestrian_txts = pedestrian_dict[scenario]
    #                 return_dict[scenario] = [
    #                     {
    #                         "labels": [str(i)],
    #                         "caption_pedestrian": pedestrian_txts[i],
    #                         "caption_vehicle": vehicle_txts[i],
    #                     } for i in label_order[scenario_idx]
    #                 ]

    #     return return_dict
