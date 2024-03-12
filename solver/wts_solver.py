import torch
import typing as tp
import math
import sys
from torchinfo import summary
from tqdm import tqdm

from .base_solver import BaseSolver
from .optimizer import adjust_learning_rate
from .formatter import Formatter
from .utils import averager

from benchmark import batch_evaluate_scenario, probe_metrics

class WTSSolver(BaseSolver):
  
    def __init__(self, cfg,
                 experiment_name,
                 signature,
                 batch_size,
                 local_dir,
                 model, 
                 train_loader,
                 val_loader,
                 optim,
                 is_eval=False,
                 hparams=None,
                 device: torch.device = torch.device("cuda")):
      
        super().__init__(cfg, signature, experiment_name, local_dir)
                
        self.device = device
        self.model = model
        self.batch_size = batch_size
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        
        if self.model.load_ckpt and not is_eval:
          checkpoint = torch.load(self.model.pretrained_ckpt, map_location="cpu")
          self.model.load_pretrained(checkpoint["model"])
          if "optimizer" in checkpoint:
              self.optim.load_state_dict(checkpoint["optimizer"])
          print("\nPretrained Vid2Seq checkpoint has beed loaded!")
          
        summary(self.model)

        self.register_stateful('model', 'optim')
        
        if cfg.LOG_TO_WANDB:
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
        
    @property
    def checkpoint_metrics_repr(self) -> tp.Optional[str]:
        if len(self.checkpoint_metrics) == 0:
            return
        prev_metrics = self.history[-1]
        if "valid" not in prev_metrics:
            return
        
        valid_metrics = prev_metrics["valid"]
        metrics_repr = [
            f"{'_'.join(k.split('/'))}_{str('{:.3f}'.format(valid_metrics[k]))}" 
            for k in self.checkpoint_metrics if k in valid_metrics
        ]
        # assert len(metrics_repr) == len(self.checkpoint_metrics)
        return "_".join(metrics_repr)

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
                vehicle_metrics = {
                    f"{loader.dataset.curr_ds_name}/vehicle_out/{k}": v 
                    for k, v in vehicle_metrics.items()
                }
                pedestrian_metrics = batch_evaluate_scenario(pred_pedestrian_text, pedestrian_text)
                pedestrian_metrics = {
                    f"{loader.dataset.curr_ds_name}/pedestrian_out/{k}": v 
                    for k, v in pedestrian_metrics.items()
                }
                
                metrics = average(
                    {f"{loader.dataset.curr_ds_name}/loss": loss, 
                     **vehicle_metrics, **pedestrian_metrics}
                )
                all_metrics = all_metrics | metrics
                lp.update(**metrics)
                
                if idx == 0:
                    self.log_text(
                        self.current_stage, 
                        [f"vehicle_pred",
                         f"vehicle_gt",
                         f"pedestrian_pred",
                         f"pedestrian_gt"],
                        [pred_vehicle_text[0], vehicle_text[0],
                         pred_pedestrian_text[0], pedestrian_text[0]],
                    )

        return all_metrics
