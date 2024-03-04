import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

from .config import get_dataset_paths
from utils import (
  get_all_top_dirs, 
  sample_files,
  simple_text_preprocess
)


class BaseDataset(Dataset):
  
  def __init__(self, cfg,
               dataset_ratios, # -1, num
               tokenizer,
               feature_branches=["vehicle"],
               random_pad_time=True):
    
    super().__init__()
    
    print(f"\n{self.__class__.__name__}'s configurations:")
    
    self.max_feats = cfg.MAX_FEATS
    self.features_dim = cfg.FEATURES_DIM
    self.max_output_tokens = cfg.MAX_OUTPUT_TOKENS
    self.max_pad_time = cfg.MAX_PAD_TIME
    self.random_pad_time = random_pad_time
    self.tokenizer = tokenizer
    
    self.sample_fps = cfg.FPS
    self.extract_fps = cfg.EXTRACT_FPS
    
    self.num_bins = cfg.NUM_BINS
    self.num_text_tokens = len(tokenizer) - self.num_bins
    
    self.feature_branches = feature_branches
    print(f"Using features: {', '.join(self.feature_branches)}.")
    
    dataset_names = [d["name"] for d in dataset_ratios]
    self.dataset_path_cfgs = get_dataset_paths(dataset_names)
    
    if len(dataset_names) > 1:
      print(f"Merging samples from: {', '.join(dataset_names)}.")
      
    self.cutoff_ds = None
    fixed_len = 0
    for idx, ds_cfg in enumerate(self.dataset_path_cfgs):
      ds_cfg["ratio"] = dataset_ratios[idx]["ratio"]
      if ds_cfg["ratio"] > 0:
        ds_cfg["all_scenarios"] = get_all_top_dirs(ds_cfg["features"])
        assert self.cutoff_ds is None
        self.cutoff_ds = ds_cfg
      else:
        ds_cfg["scenarios"] = get_all_top_dirs(ds_cfg["features"])
        ds_cfg["len"] = len(ds_cfg["scenarios"])
        fixed_len += ds_cfg["len"]
        
    if self.cutoff_ds is not None:
      assert fixed_len > 0
      self.cutoff_ds["len"] = int(fixed_len / (1 - self.cutoff_ds["ratio"])) - fixed_len
      actual_len = len(self.cutoff_ds["all_scenarios"])
      if self.cutoff_ds["len"] > actual_len:
        print(f"WARNING: {self.cutoff_ds['name']}'s cutoff length ({self.cutoff_ds['len']}) \
          is higher than actual length ({actual_len}). Setting cutoff length to {actual_len}...")
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
      self.ds_indices += [idx] * scenarios_num
      self.ds_scenario_start_indice.append(self.ds_scenario_start_indice[-1] + scenarios_num)
    assert self.ds_scenario_start_indice[-1] == len(self.ds_indices)
    
    self.ds_scenario_start_indice = self.ds_scenario_start_indice[:-1]
    print(f"Total samples: {len(self.ds_indices)}.")
    
  def _get_scenario(self, idx):
    ds_idx = self.ds_indices[idx]
    ds_cfg = self.dataset_path_cfgs[ds_idx]
    scenario = ds_cfg["scenarios"][idx - self.ds_scenario_start_indice[ds_idx]]
    return ds_cfg, scenario
  
  def _load_features(self, idx):
    ds_cfg, scenario = self._get_scenario(idx)
    is_external = ds_cfg["bbox_vehicle"] is None
    
    caption_dir = os.path.join(ds_cfg["annotations"], "captions")
    caption_path = os.path.join(caption_dir, sample_files(caption_dir)) # type: ignore
    with open(caption_path, 'r') as caption_file:
      feat_dict = json.load(caption_file)
      
    new_phases, duration, start_inc_num, end_inc_num = \
      self._get_time(feat_dict["event_phase"])
    feat_dict["event_phase"] = new_phases[::-1]
    feat_dict["duration"] = duration
    
    total_frames = sum([len(phase["all_frames"]) for phase in new_phases])
    
    if is_external:
      vehicle_dir = os.path.join(ds_cfg["features"], scenario)
    else:
      vehicle_dir = os.path.join(ds_cfg["features"], scenario, "vehicle_view") # npy
    vehicle_path = os.path.join(vehicle_dir, sample_files(vehicle_dir)) # type: ignore
    vehicle_feats = torch.from_numpy(np.load(vehicle_path)).float()
    assert len(vehicle_feats) == total_frames
    vehicle_feats = vehicle_feats[start_inc_num:] if end_inc_num == 0 \
      else vehicle_feats[start_inc_num : end_inc_num]
    feat_dict["vehicle"] = self._resample_video_features(vehicle_feats)
      
    if "overhead" in self.feature_branches:
      if is_external:
        feat_dict["overhead"] = None
      else:
        overhead_dir = os.path.join(ds_cfg["features"], scenario, "overhead_view") # npy
        overhead_path = os.path.join(overhead_dir, sample_files(overhead_dir)) # type: ignore
        overhead_feats = torch.from_numpy(np.load(overhead_path)).float()
        assert len(overhead_feats) == total_frames
        overhead_feats = overhead_feats[start_inc_num:] if end_inc_num == 0 \
          else overhead_feats[start_inc_num : end_inc_num]
        feat_dict["overhead"] = self._resample_video_features(overhead_feats)
        
    return feat_dict
  
  def _redistribute_cutoff_samples(self):
    if self.cutoff_ds is not None:
      self.cutoff_ds["scenarios"] = np.random.choice(
        self.cutoff_ds["all_scenarios"], self.cutoff_ds["len"]
      )
  
  def _get_time(self, phases):
    assert int(phases[-1]["labels"]) == 0
    max_pad_frames = self.max_pad_time * self.sample_fps
    
    true_start = float(phases[-1]["start_time"])
    true_start_frame = int(true_start * self.extract_fps)
    all_frames_start = [int(frame) for frame in phases[-1]["all_frames"]]
    start_inc_num = np.random.randint(max_pad_frames + 1) \
      if self.random_pad_time else max_pad_frames
    if all_frames_start[start_inc_num] > true_start_frame:
      start_inc_num -= 1
    assert all_frames_start[start_inc_num] <= true_start_frame
    
    true_end = float(phases[0]["end_time"])
    true_end_frame = int(true_end * self.extract_fps)
    all_frames_end = [int(frame) for frame in phases[0]["all_frames"]]
    end_inc_num = np.random.randint(max_pad_frames + 1) \
      if self.random_pad_time else max_pad_frames
    if all_frames_end[-(end_inc_num + 1)] < true_end_frame:
      end_inc_num -= 1
    assert all_frames_end[-(end_inc_num + 1)] >= true_end_frame
    
    new_start_frame = all_frames_start[start_inc_num]
    new_end_frame = all_frames_end[-(end_inc_num + 1)]
    new_start = new_start_frame / self.extract_fps
    new_end = new_end_frame / self.extract_fps
    duration = new_end - new_start
    for phase in phases:
      phase["n_start_time"] = phases["start_time"] - new_start
      phase["n_end_time"] = phases["end_time"] - new_start
    
    return phases, duration, start_inc_num, end_inc_num
  
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
    vehicle = feat_dict["vehicle"]
    overhead = feat_dict["overhead"] if "overhead" in self.feature_branches else None
    duration = feat_dict["duration"]
    phases = feat_dict["event_phase"]
    
    start = [phase["n_start_time"] for phase in phases]
    end = [phase["n_end_time"] for phase in phases]
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
    
    return {
      "vehicle": vehicle,
      "overhead": overhead,
      "output_tokens": output_tokens
    }
  
  def __len__(self):
    return len(self.ds_indices)
