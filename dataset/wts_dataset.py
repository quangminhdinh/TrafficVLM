import torch
import copy

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
                     random_pad_time=False,
                     return_raw_text=True,
                     augment=False)
    
    self.cutoff_ds = None
    
    self.all_sub_ds = []
    self.ds_cfgs_dict = {}
    for ds_cfg in self.dataset_path_cfgs:
      key = ds_cfg["name"], "vehicle"
      self.ds_cfgs_dict[key] = copy.deepcopy(ds_cfg)
      self.all_sub_ds.append(key)
      if ds_cfg["bbox_vehicle"] is not None:
        self._prune_branch(key)
        
        key = ds_cfg["name"], "overhead"
        self.ds_cfgs_dict[key] = copy.deepcopy(ds_cfg)
        self.all_sub_ds.append(key)
        self._prune_branch(key)
    
    self.reset_counter()
  
  def _prune_branch(self, key):
    name, branch = key
    ds_cfg = self.ds_cfgs_dict[key]
    eligible = []
    usable = []
    for idx, scenario in enumerate(ds_cfg["scenarios"]):
      if branch in ds_cfg["usable"][idx]:
        eligible.append(scenario)
        usable.append(ds_cfg["usable"][idx])
    removed = ds_cfg["len"] - len(eligible)
    ds_cfg["scenarios"] = eligible
    ds_cfg["usable"] = usable
    ds_cfg["len"] = len(eligible)
    print(f"- Removed {removed} sampled in {name} dataset, branch {branch}. Total: {len(eligible)} samples")
    
  def next_dataset(self):
    self.curr_ds_idx += 1
    if self.curr_ds_idx >= len(self.all_sub_ds):
      return False
    key = self.all_sub_ds[self.curr_ds_idx]
    name, branch = key
    self.curr_ds = self.ds_cfgs_dict[key]
    self.feature_branches = [branch]
    self.curr_ds_name = f"{name}_{branch}"
    print(f"Evaluating {name} dataset, {branch} branch...")
    return True
  
  def reset_counter(self):
    self.curr_ds_idx = -1
    self.curr_ds = None
    self.curr_ds_name = None
  
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
  
  vehicle_tokens = [batch[i]["vehicle_tokens"] for i in range(bs)]
  max_vehicle_len = max(len(x) for x in vehicle_tokens)
  for i in range(bs):
    if len(vehicle_tokens[i]) < max_vehicle_len:
      vehicle_tokens[i] = torch.cat([vehicle_tokens[i], 
                                    torch.zeros(max_vehicle_len - len(vehicle_tokens[i])).long()
                                    ], 0)
  vehicle_tokens = torch.stack(vehicle_tokens)
  
  pedestrian_tokens = [batch[i]["pedestrian_tokens"] for i in range(bs)]
  max_pedestrian_len = max(len(x) for x in pedestrian_tokens)
  for i in range(bs):
    if len(pedestrian_tokens[i]) < max_pedestrian_len:
      pedestrian_tokens[i] = torch.cat([pedestrian_tokens[i], 
                                    torch.zeros(max_pedestrian_len - len(pedestrian_tokens[i])).long()
                                    ], 0)
  pedestrian_tokens = torch.stack(pedestrian_tokens)
  
  if "vehicle_text" in batch[0]:
    vehicle_text = [batch[i]["vehicle_text"] for i in range(bs)]
    pedestrian_text = [batch[i]["pedestrian_text"] for i in range(bs)]
    
    return {
      "feat": feat,
      "vehicle_tokens": vehicle_tokens,
      "pedestrian_tokens": pedestrian_tokens,
      "vehicle_text": vehicle_text,
      "pedestrian_text": pedestrian_text,
    }
  
  return {
    "feat": feat,
    "vehicle_tokens": vehicle_tokens,
    "pedestrian_tokens": pedestrian_tokens
  }
