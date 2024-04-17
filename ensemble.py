import argparse
import os
import torch
import gc
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import (
  SequentialSampler,
  DataLoader
)

from args import get_test_args_parser
from config import get_cfg_defaults
from utils import fix_seed
from models import get_tokenizer, TrafficVLM
from dataset import (
  WTSTestDataset,
  wts_test_collate_fn,
)
from solver import setup_logging
from benchmark import batch_parse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_exp(exp_name):
  cfg = get_cfg_defaults()
  cfg.merge_from_file(f"experiments/{exp_name}.yml")
  cfg.freeze()
  return cfg


def batch_tokens(tokens, bs):
  max_len = max(x.shape[-1] for x in tokens)
  for i in range(bs):
    if tokens[i].shape[-1] < max_len:
      tokens[i] = torch.cat(
        [tokens[i], 
         torch.zeros(
           *tokens[i].shape[:-1], 
           max_len - tokens[i].shape[-1]
         ).to(tokens[i].device).long()], -1
      )
  return torch.stack(tokens)


def ensemble_single(model_list, tokenizer, feat, tgt_type, max_output_tokens, local_batch=None, sub_feat=None):
  all_sequences = []
  scores = []
  for model_cfg, model in model_list:
    tbu_cfg = model_cfg.SOLVER.TEST
    outputs = model.generate(
      feats=feat,
      tgt_type=tgt_type,
      local_batch=local_batch,
      sub_feats=sub_feat,
      use_nucleus_sampling=tbu_cfg.NUM_BEAMS == 0,
      num_beams=tbu_cfg.NUM_BEAMS,
      max_length=max_output_tokens,
      min_length=1,
      top_p=tbu_cfg.TOP_P if tbu_cfg.NUM_BEAMS == 0 else 1.0,
      repetition_penalty=tbu_cfg.REPETITION_PENALTY,
      length_penalty=tbu_cfg.LENGTH_PENALTY,
      num_captions=1,
      temperature=tbu_cfg.TEMPERATURE,
      output_scores=True
    )
    all_sequences.append(outputs.sequences)
    score = model.compute_reconstructed_scores(outputs, tbu_cfg.LENGTH_PENALTY)
    # if not model.use_local:
    #   for s in range(len(score)):
    #     score[s] = score[s] * 4 / 5 if score[s] < 0 else score[s] * 5 / 4
    scores.append(score)
  scores = torch.stack(scores)
  all_sequences = batch_tokens(all_sequences, len(all_sequences))
  max_idx = torch.argmax(scores, dim=0)
  final_sequences = []
  for idx, max_idx in enumerate(max_idx):
    final_sequences.append(all_sequences[max_idx][idx])
  final_sequences = torch.stack(final_sequences)
  return tokenizer.batch_decode(
    final_sequences, skip_special_tokens=True
  )
   

def generate_single(model_list, tokenizer, feat, max_output_tokens, tgt_type, scenarios, num_phases, local_batch=None, sub_feat=None):
    texts = ensemble_single(
      model_list, tokenizer, feat, tgt_type, max_output_tokens, local_batch, sub_feat
    )
    parsed_texts = batch_parse(texts)
    assert len(scenarios) == len(parsed_texts)
    
    results = {}
    for idx, scenario in enumerate(scenarios):
        scenario_texts = parsed_texts[idx]
        if len(scenario_texts) < num_phases[idx]:
            print(f"MAX TRIALS REACHED! Duplicating last text for {scenario}'s {tgt_type}...")
            dup = scenario_texts[-1]
            for _ in range(num_phases[idx] - len(scenario_texts)):
                scenario_texts.append(dup)
        results[scenario] = scenario_texts
    return results
  

def generate_all(model_list, tokenizer, loader, device):
  loader.dataset.reset_counter()
  
  return_dict = {}
  num_broken = 0
  while loader.dataset.next_dataset():
    num_samples = len(loader)
    
    for _, batch in tqdm(enumerate(loader), leave=False, total=num_samples, 
                         desc=loader.dataset.curr_ds_name):
      feat = batch["feat"].to(device)
      scenarios = batch["scenario"]
      label_order = batch["label_order"]
      num_phases = [len(label) for label in label_order]
      local_batch = batch["local"] if "local" in batch else None
      sub_feat = batch["sub_feat"].to(device) if "sub_feat" in batch else None
      
      vehicle_dict = generate_single(
        model_list,
        tokenizer,
        feat, 
        loader.dataset.max_output_tokens, 
        "vehicle", 
        scenarios, 
        num_phases, 
        local_batch, 
        sub_feat
      )
      pedestrian_dict = generate_single(
        model_list,
        tokenizer,
        feat, 
        loader.dataset.max_output_tokens, 
        "pedestrian", 
        scenarios, 
        num_phases, 
        local_batch, 
        sub_feat
      )
      for scenario_idx, scenario in enumerate(scenarios):
        assert scenario not in return_dict
        vehicle_txts = vehicle_dict[scenario]
        pedestrian_txts = pedestrian_dict[scenario]
        return_dict[scenario] = [
            {
                "labels": [str(i)],
                "caption_pedestrian": pedestrian_txts[i],
                "caption_vehicle": vehicle_txts[i],
            } for i in label_order[scenario_idx]
        ]
    if "remain" in loader.dataset.curr_ds:
        num_broken += len(loader.dataset.curr_ds["remain"])
        for remain_scenario in loader.dataset.curr_ds["remain"]:
            return_dict[remain_scenario] = return_dict[scenario]
  print("Number of broken scenarios:", num_broken)

  return return_dict


def main(args, cfg):
  torch.cuda.empty_cache()
  gc.collect()
  
  experiment_dir = os.path.join(cfg.GLOB.EXP_PARENT_DIR, args.experiment)
  setup_logging(experiment_dir)
  
  fix_seed(cfg.GLOB.SEED)
  
  exp_list = cfg.ENSEMBLE.EXPERIMENT_LIST
  assert len(exp_list) > 0
  
  device = torch.device(args.device)
  tokenizer = get_tokenizer(cfg.MODEL.T5_PATH, cfg.DATA.NUM_BINS)
  
  test_set = WTSTestDataset(cfg.DATA, tokenizer, cfg.MODEL.USE_LOCAL, cfg.MODEL.MAX_PHASES)
  
  test_sampler = SequentialSampler(test_set)
  
  test_loader = DataLoader(test_set,
                           batch_size=args.batch,
                           sampler=test_sampler,
                           collate_fn=wts_test_collate_fn,
                           num_workers=os.cpu_count()) # type: ignore
  
  model_list = []
  for exp in exp_list:
    model_cfg = load_exp(exp)
    model = TrafficVLM(
      model_cfg.MODEL, 
      tokenizer, 
      model_cfg.DATA.NUM_BINS, 
      model_cfg.DATA.MAX_FEATS, 
      model_cfg.DATA.SUB_FEATURE is not None, 
      is_eval=True
    )
    
    ckpt_parent = model_cfg.ENSEMBLE.ROOT_EXP if model_cfg.ENSEMBLE.ROOT_EXP is not None else exp
    exp_dir = Path(model_cfg.GLOB.EXP_PARENT_DIR) / ckpt_parent
    load_path = exp_dir / f'epoch_{model_cfg.SOLVER.LOAD_FROM_EPOCH}.th'
    assert load_path.exists()
    state = torch.load(load_path, 'cpu')
    model.load_state_dict(state['model'])
    
    print(f"Pretrained loaded for experiment {exp} at epoch {model_cfg.SOLVER.LOAD_FROM_EPOCH}!")
    
    model.to(device)
    model.eval()
    model_list.append((model_cfg, model))
  
  results_dict = generate_all(model_list, tokenizer, test_loader, device)
  print("Total samples:", len(results_dict))
  json_object = json.dumps(results_dict, indent=4)
  
  outdir = experiment_dir
  if not os.path.exists(outdir):
    os.makedirs(outdir)
    
  result_path = os.path.join(
    outdir, f"{args.experiment}.json"
  )
  with open(result_path, "w") as outfile:
    outfile.write(json_object)
    
  print("Test result is saved at", result_path)

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(parents=[get_test_args_parser()])
  args = parser.parse_args()
  
  # config experiment
  cfg = load_exp(args.experiment)
  
  main(args, cfg)
