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

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args, cfg):
  torch.cuda.empty_cache()
  gc.collect()
  
  experiment_dir = os.path.join(cfg.GLOB.EXP_PARENT_DIR, args.experiment)
  setup_logging(experiment_dir)
  
  fix_seed(cfg.GLOB.SEED)
  
  device = torch.device(cfg.GLOB.DEVICE)
  tokenizer = get_tokenizer(cfg.MODEL.T5_PATH, cfg.DATA.NUM_BINS)
  
  train_set = WTSTrainDataset(
    cfg.DATA, 
    tokenizer, 
    cfg.MODEL.FEATURE_BRANCHES,
    cfg.SOLVER.TRAIN.DENOISING,
    cfg.SOLVER.TRAIN.PHASE_NOISE_DENSITY,
    cfg.MODEL.USE_LOCAL,
    cfg.MODEL.MAX_PHASES
  )
  val_set = WTSValDataset(cfg.DATA, tokenizer, cfg.MODEL.USE_LOCAL, cfg.MODEL.MAX_PHASES)
  
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
  
  model = TrafficVLM(
    cfg.MODEL, tokenizer, cfg.DATA.NUM_BINS, cfg.DATA.MAX_FEATS, cfg.DATA.SUB_FEATURE is not None
  )
  model.to(device)
  
  # Set up optimizer
  params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
  optimizer = get_optimizer(cfg.SOLVER.TRAIN.OPTIMIZER, params_for_optimization)
  
  hparams = convert_to_dict(cfg)
  signature = get_sig(hparams)
  
  solver = get_solver(
    cfg.SOLVER,
    args.experiment,
    signature=signature,
    local_dir=experiment_dir,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optim=optimizer,
    hparams=hparams,
    device=device
  )
  
  solver.run()
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(parents=[get_args_parser()])
  args = parser.parse_args()
  
  # config experiment
  cfg = get_cfg_defaults()
  cfg.merge_from_file(f"experiments/{args.experiment}.yml")
  cfg.freeze()
  
  main(args, cfg)
