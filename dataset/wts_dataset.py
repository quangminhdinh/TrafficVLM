import torch

from .base_dataset import BaseDataset


class WTSTrainDataset(BaseDataset):
  
  def __init__(self, cfg, tokenizer, feature_branches=["mix"]):
    super().__init__(cfg, cfg.TRAIN_DATASETS, tokenizer,
                     feature_branches, cfg.TRAIN_RANDOM_PAD_TIME)
    

class WTSValDataset(BaseDataset):
  
  def __init__(self, cfg, tokenizer, dataset_cfgs=None):
    
    super().__init__(cfg, 
                     dataset_cfgs or cfg.VAL_DATASETS, 
                     tokenizer,
                     feature_branches=["mix"], 
                     random_pad_time=cfg.TRAIN_RANDOM_PAD_TIME,
                     return_raw_text=True)
    
    self.cutoff_ds = None
    
    self.all_sub_ds = []
    self.ds_cfgs_dict = {}
    for ds_cfg in self.dataset_path_cfgs:
      self.ds_cfgs_dict[ds_cfg["name"]] = ds_cfg
      self.all_sub_ds.append((ds_cfg["name"], "vehicle"))
      if ds_cfg["bbox_vehicle"] is not None:
        self.all_sub_ds.append((ds_cfg["name"], "overhead"))
    
    self.curr_ds_idx = -1
    self.curr_ds = None
    self.curr_ds_name = None
    
  def next_dataset(self):
    self.curr_ds_idx += 1
    if self.curr_ds_idx >= len(self.all_sub_ds):
      return False
    name, branch = self.all_sub_ds[self.curr_ds_idx]
    self.curr_ds = self.ds_cfgs_dict[name]
    self.feature_branches = [branch]
    self.curr_ds_name = f"{name}_{branch}"
    print(f"Evaluating {name} dataset, {branch} branch...")
    return True
  
  def _get_scenario(self, idx):
    assert self.curr_ds is not None
    return self.curr_ds, self.curr_ds["scenarios"][idx]
  
  def __len__(self):
    assert self.curr_ds is not None
    return len(self.curr_ds["scenarios"])
    

class WTSTestDataset(WTSValDataset):
  
  def __init__(self, cfg, tokenizer):
    super().__init__(cfg, tokenizer, cfg.TEST_DATASETS)
    
  # TODO
  def __getitem__(self, idx):
    ds_cfg, scenario = self._get_scenario(idx)
    
    feat_dict = self._load_features(idx)
    if "vehicle" in feat_dict and "overhead" in feat_dict:
      raise NotImplementedError()
    elif "vehicle" in feat_dict:
      feat = feat_dict["vehicle"]
      view = feat_dict["vehicle_view"]
    else:
      feat = feat_dict["overhead"]
      view = feat_dict["overhead_view"]

    return {
      "feat": feat,
    }


def wts_base_collate_fn(batch):
  bs = len(batch)
  feat = torch.stack([batch[i]["feat"] for i in range(bs)])
  # overhead = [batch[i]["overhead"] for i in range(bs)]
  
  output_tokens = [batch[i]["output_tokens"] for i in range(bs)]
  max_output_len = max(len(x) for x in output_tokens)
  for i in range(bs):
    if len(output_tokens[i]) < max_output_len:
      output_tokens[i] = torch.cat([output_tokens[i], 
                                    torch.zeros(max_output_len - len(output_tokens[i])).long()
                                    ], 0)
  output_tokens = torch.stack(output_tokens)
  
  if "output_text" in batch[0]:
    output_text = [batch[i]["output_text"] for i in range(bs)]
    
    return {
      "feat": feat,
      "output_tokens": output_tokens,
      "output_text": output_text,
    }
  
  return {
    "feat": feat,
    "output_tokens": output_tokens
  }
