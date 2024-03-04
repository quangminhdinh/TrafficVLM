import torch
import typing as tp
import math
import sys

from .base_solver import BaseSolver
from .optimizer import adjust_learning_rate
from .formatter import Formatter
from .utils import averager


class WTSSolver(BaseSolver):
  
    def __init__(self, cfg,
                 experiment_name,
                 signature,
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
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        
        if self.model.load_ckpt and not is_eval:
          checkpoint = torch.load(self.model.pretrained_ckpt, map_location="cpu")
          self.model.load_pretrained(checkpoint["model"])
          self.optim.load_state_dict(checkpoint["optimizer"])
          print("\nPretrained Vid2Seq checkpoint has beed loaded!")

        self.register_stateful('model', 'optim')
        self.init_wandb(
            log_folder=local_dir,
            project=experiment_name,
            signature=signature,
        )
        self.hparams = hparams
        self.max_epoch = self.train_cfg.MAX_EPOCH
        self.log_updates = self.train_cfg.LOG_UPDATES
        self.checkpoint_metrics = self.train_cfg.CHECKPOINT_METRICS
        
    @property
    def checkpoint_metrics_repr(self) -> tp.Optional[str]:
        if len(self.checkpoint_metrics) == 0:
            return
        prev_metrics = self.history[-1]
        assert "valid" in prev_metrics
        
        valid_metrics = prev_metrics["valid"]
        metrics_repr = [
            f"{k}_{str(valid_metrics[k])}" for k in self.checkpoint_metrics if k in valid_metrics
        ]
        assert len(metrics_repr) == len(self.checkpoint_metrics)
        return "_".join(metrics_repr)

    def run(self):
        self.logger.info('Log dir: %s', self.folder)
        self.restore()
        if self.hparams is not None:
            self.log_hyperparams(self.hparams)
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.run_stage("train", self.do_train)
            self.run_stage("valid", self.do_train_valid, train=False)
            self.commit()

    def get_formatter(self, stage_name: str):
        if stage_name == "train":
            return Formatter({
                'loss': '.5f',
                'lr': 'e',
            })
        elif stage_name == "valid":
            return Formatter({
                'loss': '.5f',
            })
        raise NotImplementedError()

    def do_train(self):
        self.logger.info('-' * 80)
        self.logger.info(f'Starting {self.current_stage} stage...')
        loader = self.train_loader
        lp = self.log_progress(
            self.current_stage, loader, total=len(loader), updates=self.log_updates
        )
        average = averager()
        self.model.train()
        num_training_steps = int(len(loader) * self.max_epoch)

        for idx, batch in enumerate(lp):
            vehicle = batch["vehicle"].to(self.device)
            overhead = batch["overhead"].to(self.device)
            output_tokens = batch["output_tokens"].to(self.device)
            
            loss = self.model(vehicle, overhead, output_tokens)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            
            self.optim.zero_grad()
            loss.backward()
            if self.train_cfg.CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_( # type: ignore
                    self.model.parameters(), self.train_cfg.CLIP_MAX_NORM
                )
            self.optim.step()
            
            with torch.no_grad():
                self.model.normalize_time_embeddings()
            adjust_learning_rate(
                self.optim,
                curr_step=(self.epoch - 1) * len(loader) + idx,
                num_training_steps=num_training_steps,
                cfg=self.optim_cfg,
            )
            
            metrics = average({'loss': loss, 'lr': self.optim.param_groups[0]["lr"]})
            lp.update(**metrics)

        return metrics
