import argparse
import os
import torch
import gc
from torch.utils.data import (
  RandomSampler,
  SequentialSampler,
  DataLoader
)

from args import get_args_parser
from config import (
  get_cfg_defaults,
  convert_to_dict,
  get_sig
)
from utils import fix_seed
from models import get_tokenizer, TrafficVLM
from dataset import (
  WTSTrainDataset,
  WTSValDataset,
  wts_base_collate_fn
)
from solver import (
  get_optimizer,
  get_solver,
  setup_logging
)


def main(args, cfg):
  torch.cuda.empty_cache()
  gc.collect()
  
  experiment_dir = os.path.join(cfg.GLOB.EXP_PARENT_DIR, args.experiment)
  setup_logging(experiment_dir)
  
  fix_seed(cfg.GLOB.SEED)
  
  device = torch.device(cfg.GLOB.DEVICE)
  tokenizer = get_tokenizer(cfg.MODEL.T5_PATH, cfg.DATA.NUM_BINS)
  
  
  cfg.DATA.VAL_DATASETS = [
    { "name": "wts_val_main", "ratio": -1 },
  ]
  hparams = convert_to_dict(cfg)
  signature = get_sig(hparams)
  
  train_set = WTSTrainDataset(cfg.DATA, tokenizer, cfg.MODEL.FEATURE_BRANCHES)
  val_set = WTSValDataset(cfg.DATA, tokenizer)
  
  train_sampler = RandomSampler(train_set)
  val_sampler = SequentialSampler(val_set)
  
  train_loader = DataLoader(train_set,
                            batch_size=cfg.SOLVER.TRAIN.BATCH_SIZE,
                            sampler=train_sampler,
                            collate_fn=wts_base_collate_fn,
                            num_workers=os.cpu_count()) # type: ignore
  val_loader = DataLoader(val_set,
                          batch_size=cfg.SOLVER.VAL.BATCH_SIZE or cfg.SOLVER.TRAIN.BATCH_SIZE,
                          sampler=val_sampler,
                          collate_fn=wts_base_collate_fn,
                          num_workers=os.cpu_count()) # type: ignore
  
  model = TrafficVLM(cfg.MODEL, tokenizer, cfg.DATA.NUM_BINS, cfg.DATA.MAX_FEATS)
  model.to(device)
  
  # Set up optimizer
  params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
  optimizer = get_optimizer(cfg.SOLVER.TRAIN.OPTIMIZER, params_for_optimization)
  
  solver = get_solver(
    cfg.SOLVER,
    args.experiment,
    signature=signature,
    batch_size=cfg.SOLVER.TRAIN.BATCH_SIZE,
    local_dir=experiment_dir,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optim=optimizer,
    hparams=hparams,
    device=device
  )
  
  # solver.run()
  for ep in range(20, 291, 10):
    solver.dm = f"/home/logs/default/epoch_{ep}_.th" # type: ignore
    solver.load_from_epoch = ep
    solver.OVERWRITE_EPOCH = ep # type: ignore
    solver.restore()
    solver.run_stage("valid", solver.do_valid)
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(parents=[get_args_parser()])
  args = parser.parse_args()
  
  # config experiment
  cfg = get_cfg_defaults()
  cfg.merge_from_file(f"experiments/{args.experiment}.yml")
  # cfg.freeze()
  
  main(args, cfg)
