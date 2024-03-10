import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

from .config import get_dataset_paths
from utils import (
  get_all_top, 
  sample_files,
  simple_text_preprocess
)


class BaseDataset(Dataset):
  
  def __init__(self, cfg,
               dataset_ratios, # -1, num
               tokenizer,
               feature_branches=["vehicle"],
               random_pad_time=True,
               return_raw_text=False,):
    
    super().__init__()
    
    print(f"\n{self.__class__.__name__}'s configurations:")
    
    self.max_feats = cfg.MAX_FEATS
    self.features_dim = cfg.FEATURES_DIM
    self.max_output_tokens = cfg.MAX_OUTPUT_TOKENS
    self.max_pad_time = cfg.MAX_PAD_TIME
    self.random_pad_time = random_pad_time
    self.tokenizer = tokenizer
    
    self.return_raw_text = return_raw_text
    
    self.sample_fps = cfg.FPS
    self.extract_fps = cfg.EXTRACT_FPS
    
    self.overhead_ratio = cfg.OVERHEAD_RATIO
    
    self.num_bins = cfg.NUM_BINS
    self.num_text_tokens = len(tokenizer) - self.num_bins
    
    self.feature_branches = feature_branches
    if self.feature_branches is not None:
      print(f"Using features: {', '.join(self.feature_branches)}.")
    
    dataset_names = [d["name"] for d in dataset_ratios]
    self.dataset_path_cfgs = get_dataset_paths(*dataset_names)
    
    if len(dataset_names) > 1:
      print(f"Merging samples from: {', '.join(dataset_names)}.")
      
    self.cutoff_ds = None
    fixed_len = 0
    for idx, ds_cfg in enumerate(self.dataset_path_cfgs):
      ds_cfg["ratio"] = dataset_ratios[idx]["ratio"]
      scenarios = get_all_top(ds_cfg["features"])
      scenarios.sort()
      if ds_cfg["ratio"] > 0:
        ds_cfg["all_scenarios"] = scenarios
        ds_cfg["broken"] = []
        assert self.cutoff_ds is None
        self.cutoff_ds = ds_cfg
      else:
        ds_cfg["scenarios"] = scenarios
        self._perform_integrity_check(ds_cfg)
        fixed_len += ds_cfg["len"]
        
    if self.cutoff_ds is not None:
      assert fixed_len > 0
      self.cutoff_ds["len"] = int(fixed_len / (1 - self.cutoff_ds["ratio"])) - fixed_len
      actual_len = len(self.cutoff_ds["all_scenarios"])
      if self.cutoff_ds["len"] > actual_len:
        print(f"WARNING: {self.cutoff_ds['name']}'s cutoff length ({self.cutoff_ds['len']}) "
          "is higher than actual length ({actual_len}). Setting cutoff length to {actual_len}...")
        self.cutoff_ds["len"] = actual_len
        self.cutoff_ds["scenarios"] = self.cutoff_ds["all_scenarios"]
        self.cutoff_ds = None
      self._redistribute_cutoff_samples()
      
    self.ds_indices = []
    self.ds_scenario_start_indice = [0]
    print("Number of samples of:")
    for idx, ds_cfg in enumerate(self.dataset_path_cfgs):
      scenarios_num = len(ds_cfg["scenarios"])
      print(f"- {ds_cfg['name']}: {scenarios_num}.")
      if len(ds_cfg["broken"]) > 0:
        print("Removing", len(ds_cfg["broken"]), "corrupted samples:")
        for broken_scenario in ds_cfg["broken"]:
          print(f"\t- {ds_cfg['name']}, {broken_scenario}.")
      else:
        print()
      self.ds_indices += [idx] * scenarios_num
      self.ds_scenario_start_indice.append(self.ds_scenario_start_indice[-1] + scenarios_num)
    assert self.ds_scenario_start_indice[-1] == len(self.ds_indices)
    
    self.ds_scenario_start_indice = self.ds_scenario_start_indice[:-1]
    print(f"Total samples: {len(self.ds_indices)}.")
    
  def _get_scenario(self, idx):
    ds_idx = self.ds_indices[idx]
    ds_cfg = self.dataset_path_cfgs[ds_idx]
    scenario = ds_cfg["scenarios"][idx - self.ds_scenario_start_indice[ds_idx]].strip()
    return ds_cfg, scenario
  
  # def _perform_external_integrity_check(self, ds_cfg):
  #   eligible = []
  #   broken = []
  #   for scenario in ds_cfg["scenarios"]
  
  def _perform_integrity_check(self, ds_cfg):
    if ds_cfg["bbox_vehicle"] is None:
      ds_cfg["broken"] = []
      ds_cfg["len"] = len(ds_cfg["scenarios"])
      return
    eligible = []
    broken = []
    ds_cfg["usable"] = []
    for scenario in ds_cfg["scenarios"]:
      usable = ["vehicle", "overhead"]
      if "vehicle" in self.feature_branches or "mix" in self.feature_branches:
        local_dir = os.path.join(ds_cfg["features"], scenario, "vehicle_view")
        if len(os.listdir(local_dir)) == 0:
          usable.remove("vehicle")
      if "overhead" in self.feature_branches or "mix" in self.feature_branches:
        local_dir = os.path.join(ds_cfg["features"], scenario, "overhead_view")
        if len(os.listdir(local_dir)) == 0:
          usable.remove("overhead")
      if len(usable) == 0:
        broken.append(scenario)
      else:
        eligible.append(scenario)
        ds_cfg["usable"].append(usable)
    if "broken" not in ds_cfg:
      ds_cfg["broken"] = []
    ds_cfg["broken"] += broken
    ds_cfg["scenarios"] = eligible
    ds_cfg["len"] = len(eligible)
  
  def _load_view_caption(self, ds_cfg, scenario, view, feat_dict):
    caption_dir = os.path.join(ds_cfg["captions"], scenario, view)
    caption_path = os.path.join(caption_dir, sample_files(caption_dir)) # type: ignore
    with open(caption_path, 'r') as caption_file:
      cap = json.load(caption_file)
    feat_dict[view] = cap
  
  def _load_caption(self, ds_cfg, scenario, is_external, feat_dict={}):
    if is_external:
      caption_path = os.path.join(ds_cfg["captions"], f"{scenario}_caption.json")
      with open(caption_path, 'r') as caption_file:
        cap = json.load(caption_file)
      feat_dict["vehicle_view"] = cap
      return
    if len(ds_cfg["usable"]) == 1 and \
      (ds_cfg["usable"][0] in self.feature_branches or "mix" in self.feature_branches):
      view = f"{ds_cfg['usable'][0]}_view"
      self._load_view_caption(ds_cfg, scenario, view, feat_dict)
      return
    if "mix" in self.feature_branches:
      view = "overhead_view" if np.random.uniform() < self.overhead_ratio else "vehicle_view"
      self._load_view_caption(ds_cfg, scenario, view, feat_dict)
      return
    if "vehicle" in self.feature_branches:
      self._load_view_caption(ds_cfg, scenario, "vehicle_view", feat_dict)
    if "overhead" in self.feature_branches:
      self._load_view_caption(ds_cfg, scenario, "overhead_view", feat_dict)
      
  def _load_clip_features(self, path, start, end, total_length):
    feats = torch.from_numpy(np.load(path)).float()
    assert len(feats) == total_length
    feats = feats[start : end]
    return self._resample_video_features(feats)
  
  def _load_features(self, idx):
    ds_cfg, scenario = self._get_scenario(idx)
    is_external = ds_cfg["bbox_vehicle"] is None
    if is_external:
      scenario = scenario.split(".")
      assert len(scenario) == 2
      scenario = scenario[0]
    
    feat_dict = {}
    self._load_caption(ds_cfg, scenario, is_external, feat_dict)
    
    if "vehicle_view" in feat_dict:
      view = feat_dict["vehicle_view"]
      all_frames = sorted(view["all_frames"])
      start_inc_num, end_inc_num = self._get_time(view, all_frames)
    
      if is_external:
        vehicle_path = os.path.join(ds_cfg["features"], f"{scenario}.npy")
      else:
        vehicle_dir = os.path.join(ds_cfg["features"], scenario, "vehicle_view") # npy
        vehicle_path = os.path.join(vehicle_dir, sample_files(vehicle_dir)) # type: ignore
      feat_dict["vehicle"] = self._load_clip_features(
        vehicle_path, start_inc_num, end_inc_num, len(all_frames)
      )
      
    if "overhead_view" in feat_dict:
      view = feat_dict["overhead_view"]
      all_frames = sorted(view["all_frames"])
      start_inc_num, end_inc_num = self._get_time(view, all_frames)
    
      overhead_dir = os.path.join(ds_cfg["features"], scenario, "overhead_view") # npy
      overhead_path = os.path.join(overhead_dir, sample_files(overhead_dir)) # type: ignore
      feat_dict["overhead"] = self._load_clip_features(
        overhead_path, start_inc_num, end_inc_num, len(all_frames)
      )

    return feat_dict
  
  def _redistribute_cutoff_samples(self):
    if self.cutoff_ds is not None:
      self.cutoff_ds["scenarios"] = np.random.choice(
        self.cutoff_ds["all_scenarios"], self.cutoff_ds["len"]
      )
  
  def _resample_video_features(self, feats):
    if len(feats) > self.max_feats:
      sampled = []
      for j in range(self.max_feats):
        sampled.append(feats[(j * len(feats)) // self.max_feats])
      feats = torch.stack(sampled)
    elif len(feats) < self.max_feats:
      video_len = len(feats)
      feats = torch.cat(
        [feats, torch.zeros(self.max_feats - video_len, self.features_dim)], 0
      )
    return feats
  
  def time_tokenize(self, x, duration, num_bins):
    time_token = int(float((num_bins - 1) * x) / float(duration))
    assert time_token <= self.num_bins
    return time_token + self.num_text_tokens
  
  def __getitem__(self, idx):
    if idx == 0:
      self._redistribute_cutoff_samples()
      
    feat_dict = self._load_features(idx)
    if "vehicle" in feat_dict and "overhead" in feat_dict:
      raise NotImplementedError()
    elif "vehicle" in feat_dict:
      feat = feat_dict["vehicle"]
      view = feat_dict["vehicle_view"]
    else:
      feat = feat_dict["overhead"]
      view = feat_dict["overhead_view"]
    duration = view["duration"]
    phases = view["event_phase"]
    
    start = [phase["start_time"] for phase in phases]
    end = [phase["end_time"] for phase in phases]
    text = [simple_text_preprocess(
      f"pedestrian: {phase['caption_pedestrian']} vehicle: {phase['caption_vehicle']}"
    ) for phase in phases]
    time_output_tokens = [torch.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                            self.time_tokenize(ed, duration, self.num_bins)])
                          for st, ed in zip(start, end)]
    text_output_tokens = [self.tokenizer(x, 
                                         add_special_tokens=False, 
                                         max_length=self.max_output_tokens,
                                         padding="do_not_pad",
                                         truncation=True,
                                         return_tensors="pt")['input_ids'][0]
                          for x in text]
    output_tokens = [torch.cat([ti, te], 0) 
                     for ti, te in zip(time_output_tokens, text_output_tokens)]
    output_tokens = torch.cat(output_tokens, 0)
    output_tokens = output_tokens[:self.max_output_tokens - 1]
    output_tokens = torch.cat([output_tokens, 
                               torch.LongTensor([self.tokenizer.eos_token_id])], 0)
    
    if not self.return_raw_text:
      return {
        "feat": feat,
        "output_tokens": output_tokens
      }
    
    output_text = self.tokenizer.batch_decode(
      output_tokens, skip_special_tokens=True
    )
    
    return {
      "feat": feat,
      "output_tokens": output_tokens,
      "output_text": output_text,
    }
  
  def __len__(self):
    return len(self.ds_indices)
  
  def _get_time(self, view, all_frames):
    phases = view["event_phase"]
    
    for phase in phases:
      phase["start_time"] = float(phase["start_time"])
      phase["end_time"] = float(phase["end_time"])
    
    if phases[-1]["end_time"] < phases[0]["start_time"]:
      phases = phases[::-1]
    
    start_frame = int(phases[0]["start_time"] * self.extract_fps)
    for idx, frame in enumerate(all_frames):
      if frame > start_frame:
        start_frame = all_frames[idx - 1] 
        phases[0]["start_time"] = start_frame / self.extract_fps
        start_inc_num = idx - 1
        break
      
    end_frame = int(phases[-1]["end_time"] * self.extract_fps)
    for idx in range(len(all_frames) - 1, -1, -1):
      if all_frames[idx] < end_frame:
        end_frame = all_frames[idx + 1] 
        phases[-1]["end_time"] = end_frame / self.extract_fps
        end_inc_num = idx + 1
        break
      
    duration = phases[-1]["end_time"] - phases[0]["start_time"]
    assert duration > 0
    
    view["event_phase"] = phases
    view["duration"] = duration
      
    return start_inc_num, end_inc_num
  
  # LEGACY
  # def _get_time(self, phases):
  #   assert int(phases[-1]["labels"]) == 0
  #   max_pad_frames = self.max_pad_time * self.sample_fps
    
  #   true_start = float(phases[-1]["start_time"])
  #   true_start_frame = int(true_start * self.extract_fps)
  #   all_frames_start = [int(frame) for frame in phases[-1]["all_frames"]]
  #   start_inc_num = np.random.randint(max_pad_frames + 1) \
  #     if self.random_pad_time else max_pad_frames
  #   if all_frames_start[start_inc_num] > true_start_frame:
  #     start_inc_num -= 1
  #   assert all_frames_start[start_inc_num] <= true_start_frame
    
  #   true_end = float(phases[0]["end_time"])
  #   true_end_frame = int(true_end * self.extract_fps)
  #   all_frames_end = [int(frame) for frame in phases[0]["all_frames"]]
  #   end_inc_num = np.random.randint(max_pad_frames + 1) \
  #     if self.random_pad_time else max_pad_frames
  #   if all_frames_end[-(end_inc_num + 1)] < true_end_frame:
  #     end_inc_num -= 1
  #   assert all_frames_end[-(end_inc_num + 1)] >= true_end_frame
    
  #   new_start_frame = all_frames_start[start_inc_num]
  #   new_end_frame = all_frames_end[-(end_inc_num + 1)]
  #   new_start = new_start_frame / self.extract_fps
  #   new_end = new_end_frame / self.extract_fps
  #   duration = new_end - new_start
  #   for phase in phases:
  #     phase["n_start_time"] = phases["start_time"] - new_start
  #     phase["n_end_time"] = phases["end_time"] - new_start
    
  #   return phases, duration, start_inc_num, end_inc_num
