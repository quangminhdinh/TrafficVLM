import torch

from .base_dataset import BaseDataset


class WTSTrainDataset(BaseDataset):
  
  def __init__(self, cfg, tokenizer, feature_branches=["vehicle"]):
    super().__init__(cfg, cfg.TRAIN_DATASETS, tokenizer,
                     feature_branches, cfg.TRAIN_RANDOM_PAD_TIME)
    

class WTSValDataset(BaseDataset):
  
  def __init__(self, cfg, tokenizer, feature_branches=["vehicle"]):
    super().__init__(cfg, cfg.VAL_DATASETS, tokenizer,
                     feature_branches, cfg.TRAIN_RANDOM_PAD_TIME)
    

def wts_base_collate_fn(batch):
  bs = len(batch)
  vehicle = torch.stack([batch[i]["vehicle"] for i in range(bs)])
  overhead = [batch[i]["overhead"] for i in range(bs)]
  
  output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
  max_output_len = max(len(x) for x in output_tokens)
  for i in range(bs):
    if len(output_tokens[i]) < max_output_len:
      output_tokens[i] = torch.cat([output_tokens[i], 
                                    torch.zeros(max_output_len - len(output_tokens[i])).long()
                                    ], 0)
  output_tokens = torch.stack(output_tokens)
  
  return {
    "vehicle": vehicle,
    "overhead": overhead,
    "output_tokens": output_tokens
  }
