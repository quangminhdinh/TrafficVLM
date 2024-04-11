import argparse
import os
import torch
import gc
import json
from torch.utils.data import (
  SequentialSampler,
  DataLoader
)

from args import get_test_args_parser
from config import (
  get_cfg_defaults,
  convert_to_dict,
  get_sig
)
from utils import fix_seed
from models import get_tokenizer, TrafficVLM
from dataset import (
  WTSTestDataset,
  wts_test_collate_fn,
)
from solver import (
  get_solver,
  setup_logging,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args, cfg):
  torch.cuda.empty_cache()
  gc.collect()
  
  experiment_dir = os.path.join(cfg.GLOB.EXP_PARENT_DIR, args.experiment)
  setup_logging(experiment_dir)
  
  fix_seed(cfg.GLOB.SEED)
  
  device = torch.device(args.device)
  tokenizer = get_tokenizer(cfg.MODEL.T5_PATH, cfg.DATA.NUM_BINS)
  
  test_set = WTSTestDataset(cfg.DATA, tokenizer, cfg.MODEL.USE_LOCAL, cfg.MODEL.MAX_PHASES)
  
  test_sampler = SequentialSampler(test_set)
  
  test_loader = DataLoader(test_set,
                           batch_size=args.batch,
                           sampler=test_sampler,
                           collate_fn=wts_test_collate_fn,
                           num_workers=os.cpu_count()) # type: ignore
  
  model = TrafficVLM(
    cfg.MODEL, tokenizer, cfg.DATA.NUM_BINS, cfg.DATA.MAX_FEATS, cfg.DATA.SUB_FEATURE is not None, is_eval=True
  )
  model.to(device)

  hparams = convert_to_dict(cfg)
  signature = get_sig(hparams)
  
  solver = get_solver(
    cfg.SOLVER,
    args.experiment,
    signature=signature,
    local_dir=experiment_dir,
    model=model,
    is_eval=True,
    device=device
  )
  
  results_dict = solver.do_test(test_loader)
  print("Total samples:", len(results_dict))
  json_object = json.dumps(results_dict, indent=4)
  
  outdir = os.path.join(solver.folder, "test_results")
  if not os.path.exists(outdir):
    os.makedirs(outdir)
    
  result_path = os.path.join(
    outdir, f"{args.experiment}_epoch_{solver.load_from_epoch}.json"
  )
  with open(result_path, "w") as outfile:
    outfile.write(json_object)
    
  print("Test result is saved at", result_path)
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(parents=[get_test_args_parser()])
  args = parser.parse_args()
  
  # config experiment
  cfg = get_cfg_defaults()
  cfg.merge_from_file(f"experiments/{args.experiment}.yml")
  cfg.freeze()
  
  main(args, cfg)
