import torch
import copy
import os
import json

from .base_dataset import BaseDataset


class WTSTrainDataset(BaseDataset):
  
  def __init__(self, 
               cfg, 
               tokenizer, 
               feature_branches=["mix"],
               denoising=False,
               phase_noise_density=0.5,
               use_local=False,
               max_phases=6):
    
    super().__init__(cfg, 
                     cfg.TRAIN_DATASETS, 
                     tokenizer,
                     feature_branches, 
                     cfg.TRAIN_RANDOM_PAD_TIME,
                     denoising=denoising,
                     phase_noise_density=phase_noise_density,
                     use_local=use_local,
                     max_phases=max_phases)
    

class WTSValDataset(BaseDataset):
  
  def __init__(self, cfg, tokenizer, use_local=False, max_phases=6):
    
    super().__init__(cfg, 
                     cfg.VAL_DATASETS, 
                     tokenizer,
                     feature_branches=["mix"], 
                     random_pad_time=False,
                     return_raw_text=True,
                     augment=False,
                     use_local=use_local,
                     max_phases=max_phases)
    
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
    return self.curr_ds, self.curr_ds["scenarios"][idx], idx
  
  def __len__(self):
    assert self.curr_ds is not None
    return len(self.curr_ds["scenarios"])
    

class WTSTestDataset(BaseDataset):
  
  def __init__(self, cfg, tokenizer, use_local=False, max_phases=6):
    
    super().__init__(cfg, 
                     cfg.TEST_DATASETS, 
                     tokenizer,
                     feature_branches=["mix"], 
                     random_pad_time=False,
                     return_raw_text=True,
                     augment=False,
                     use_local=use_local,
                     max_phases=max_phases)
    
    self.cutoff_ds = None
    
    self.all_sub_ds = []
    self.ds_cfgs_dict = {}
    for ds_cfg in self.dataset_path_cfgs:
      self.ds_cfgs_dict[ds_cfg["name"]] = ds_cfg
      self.all_sub_ds.append(ds_cfg["name"])
    
    self.reset_counter()

  def next_dataset(self):
    self.curr_ds_idx += 1
    if self.curr_ds_idx >= len(self.all_sub_ds):
      self.reset_counter()
      return False
    self.curr_ds_name = self.all_sub_ds[self.curr_ds_idx]
    self.curr_ds = self.ds_cfgs_dict[self.curr_ds_name]
    assert len(self.curr_ds["broken"]) == 0
    print(f"Generating samples for {self.curr_ds_name} dataset...")
    return True
  
  def reset_counter(self):
    self.curr_ds_idx = -1
    self.curr_ds = None
    self.curr_ds_name = None
  
  def _get_scenario(self, idx):
    assert self.curr_ds is not None
    return self.curr_ds, self.curr_ds["scenarios"][idx], idx
  
  def __len__(self):
    assert self.curr_ds is not None
    return len(self.curr_ds["scenarios"])
  
  def _load_caption(self, ds_cfg, usable, scenario, is_external, feat_dict={}):
    if is_external:
      caption_path = os.path.join(ds_cfg["captions"], f"{scenario}_caption.json")
      with open(caption_path, 'r') as caption_file:
        cap = json.load(caption_file)
      feat_dict["view"] = cap
      feat_dict["branch"] = "vehicle"
      return
    assert len(usable) > 0
    if len(usable) == 1:
      self._load_view_caption(ds_cfg, scenario, usable[0], feat_dict)
      return
    self._load_view_caption(ds_cfg, scenario, "vehicle", feat_dict)

  def __getitem__(self, idx):
    ds_cfg, scenario, _ = self._get_scenario(idx)
    is_external = ds_cfg["bbox_vehicle"] is None
    if is_external:
      scenario = scenario.split(".")
      assert len(scenario) == 2
      scenario = scenario[0]
    
    feat_dict = self._load_features(idx)
    
    feat = feat_dict["feats"]
    view = feat_dict["view"]

    ret = {
      "feat": feat,
      "scenario": scenario,
      "label_order": [int(l) for l in view["label_order"]],
    }
    if self.use_local:
      ret["local"] = view["local"]
      
    if self.sub_feature is not None:
      ret["sub_feat"] = feat_dict["sub_feats"]

    return ret


def wts_test_collate_fn(batch):
  bs = len(batch)
  feat = torch.stack([batch[i]["feat"] for i in range(bs)])
  scenario = [batch[i]["scenario"] for i in range(bs)]
  label_order = [batch[i]["label_order"] for i in range(bs)]
  
  ret = {
    "feat": feat,
    "scenario": scenario,
    "label_order": label_order,
  }
  
  if "local" in batch[0]:
    ret["local"] = [batch[i]["local"] for i in range(bs)]
  if "sub_feat" in batch[0]:
    ret["sub_feat"] = torch.stack([batch[i]["sub_feat"] for i in range(bs)])

  return ret
  

def batch_tokens(tokens, bs):
  max_len = max(len(x) for x in tokens)
  for i in range(bs):
    if len(tokens[i]) < max_len:
      tokens[i] = torch.cat(
        [tokens[i], torch.zeros(max_len - len(tokens[i])).long()], 0
      )
  return torch.stack(tokens)


def wts_base_collate_fn(batch):
  bs = len(batch)
  feat = torch.stack([batch[i]["feat"] for i in range(bs)])
  
  vehicle_tokens = [batch[i]["vehicle_tokens"] for i in range(bs)]
  vehicle_tokens = batch_tokens(vehicle_tokens, bs)
  
  pedestrian_tokens = [batch[i]["pedestrian_tokens"] for i in range(bs)]
  pedestrian_tokens = batch_tokens(pedestrian_tokens, bs)
  
  ret: dict = {
    "feat": feat,
    "vehicle_tokens": vehicle_tokens,
    "pedestrian_tokens": pedestrian_tokens
  }
  
  if "local" in batch[0]:
    ret["local"] = [batch[i]["local"] for i in range(bs)]
    
  if "sub_feat" in batch[0]:
    ret["sub_feat"] = torch.stack([batch[i]["sub_feat"] for i in range(bs)])
  
  if "vehicle_text" in batch[0]:
    vehicle_text = [batch[i]["vehicle_text"] for i in range(bs)]
    pedestrian_text = [batch[i]["pedestrian_text"] for i in range(bs)]
    
    ret["vehicle_text"] = vehicle_text 
    ret["pedestrian_text"] = pedestrian_text
  
  if "denoising_feat" in batch[0]:
    denoising_feat = torch.stack([batch[i]["denoising_feat"] for i in range(bs)])
    
    denoise_vehicle_tokens = [batch[i]["denoise_vehicle_tokens"] for i in range(bs)]
    denoise_vehicle_tokens = batch_tokens(denoise_vehicle_tokens, bs)
    
    denoise_pedestrian_tokens = [batch[i]["denoise_pedestrian_tokens"] for i in range(bs)]
    denoise_pedestrian_tokens = batch_tokens(denoise_pedestrian_tokens, bs)
    
    ret["denoising_feat"] = denoising_feat
    ret["denoise_vehicle_tokens"] = denoise_vehicle_tokens
    ret["denoise_pedestrian_tokens"] = denoise_pedestrian_tokens
  
  return ret
