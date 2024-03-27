import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

from .config import get_dataset_paths
from utils import (
  get_all_top, 
  sample_files,
  simple_text_preprocess,
  sample_every,
  phase_sentinel_text_mask,
  phase_sentinel_vid_mask,
  Augmentor
)


class BaseDataset(Dataset):
  
  def __init__(self, cfg,
               dataset_ratios, # -1, num
               tokenizer,
               feature_branches=["vehicle"],
               random_pad_time=True,
               return_raw_text=False,
               augment=True,
               denoising=False,
               phase_noise_density=0.5,
               use_local=False,
               max_phases=6):
    
    super().__init__()
    
    print(f"\n{self.__class__.__name__}'s configurations:")
    
    self.max_feats = cfg.MAX_FEATS
    self.features_dim = cfg.FEATURES_DIM
    self.max_output_tokens = cfg.MAX_OUTPUT_TOKENS
    self.max_pad_time = cfg.MAX_PAD_TIME
    self.random_pad_time = random_pad_time
    if self.random_pad_time:
      print(f"0 to {self.max_pad_time}s will be added to the true feature segment.")
    self.tokenizer = tokenizer
    
    self.return_raw_text = return_raw_text
    
    self.sample_fps = cfg.FPS
    self.extract_fps = cfg.EXTRACT_FPS
    
    self.overhead_ratio = cfg.OVERHEAD_RATIO
    
    self.num_bins = cfg.NUM_BINS
    self.num_text_tokens = len(tokenizer) - self.num_bins
    
    self.denoising = denoising
    self.phase_noise_density = phase_noise_density
    if self.denoising:
      print("Phase noises will be added for denoising training with density",
            self.phase_noise_density)
    
    self.feature_branches = feature_branches
    if self.feature_branches is not None:
      print(f"Using features: {', '.join(self.feature_branches)}.")
    
    self.use_local = use_local
    self.max_phases = max_phases
    
    self.main_feature = cfg.MAIN_FEATURE
    self.sub_feature = cfg.SUB_FEATURE
    print("Main feature:", self.main_feature)
    print("Sub feature:", self.sub_feature)
    
    dataset_names = [d["name"] for d in dataset_ratios]
    self.dataset_path_cfgs = get_dataset_paths(*dataset_names)
    
    if len(dataset_names) > 1:
      print(f"Merging samples from: {', '.join(dataset_names)}.")
      
    self.cutoff_ds = None
    fixed_len = 0
    for idx, ds_cfg in enumerate(self.dataset_path_cfgs):
      ds_cfg["ratio"] = dataset_ratios[idx]["ratio"]
      scenarios = get_all_top(ds_cfg[self.main_feature])
      if ds_cfg["bbox_vehicle"] is None:
        cap_side = get_all_top(ds_cfg["captions"])
        if len(cap_side) > len(scenarios):
          cap_side = [f"{cap.split('_')[0]}.npy" for cap in cap_side]
          ds_cfg["remain"] = list(set(cap_side) - set(scenarios))
          ds_cfg["remain"] = [scen[:-4] for scen in ds_cfg["remain"]]
      if "normal_trimmed" in scenarios:
        print(f"REMOVING {'normal_trimmed'.upper()} from {ds_cfg['name'].upper()}...")
        scenarios.remove("normal_trimmed")
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
    
    self.augmentor = Augmentor(
      cfg.AUGMENT,
      cfg.AUGMENT.NLP_AUGS if augment else []
    )
    
  def _get_scenario(self, idx):
    ds_idx = self.ds_indices[idx]
    ds_cfg = self.dataset_path_cfgs[ds_idx]
    in_ds_idx = idx - self.ds_scenario_start_indice[ds_idx]
    scenario = ds_cfg["scenarios"][in_ds_idx].strip()
    return ds_cfg, scenario, in_ds_idx
  
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
        local_dir = os.path.join(ds_cfg[self.main_feature], scenario, "vehicle_view")
        if len(os.listdir(local_dir)) == 0:
          usable.remove("vehicle")
      else:
        usable.remove("vehicle")
      if "overhead" in self.feature_branches or "mix" in self.feature_branches:
        local_dir = os.path.join(ds_cfg[self.main_feature], scenario, "overhead_view")
        if len(os.listdir(local_dir)) == 0:
          usable.remove("overhead")
      else:
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
  
  def _load_view_caption(self, ds_cfg, scenario, branch, feat_dict):
    caption_dir = os.path.join(ds_cfg["captions"], scenario, f"{branch}_view")
    caption_path = os.path.join(caption_dir, sample_files(caption_dir)) # type: ignore
    with open(caption_path, 'r') as caption_file:
      cap = json.load(caption_file)
    feat_dict["view"] = cap
    feat_dict["branch"] = branch
  
  def _load_caption(self, ds_cfg, usable, scenario, is_external, feat_dict={}):
    if is_external:
      caption_path = os.path.join(ds_cfg["captions"], f"{scenario}_caption.json")
      with open(caption_path, 'r') as caption_file:
        cap = json.load(caption_file)
      feat_dict["view"] = cap
      feat_dict["branch"] = "vehicle"
      return
    if len(usable) == 1 and (usable[0] in self.feature_branches or "mix" in self.feature_branches):
      self._load_view_caption(ds_cfg, scenario, usable[0], feat_dict)
      return
    if len(usable) == 1:
      raise RuntimeError("Incorrect behaviour!")
    if "mix" in self.feature_branches:
      branch = "overhead" if np.random.uniform() < self.overhead_ratio else "vehicle"
      self._load_view_caption(ds_cfg, scenario, branch, feat_dict)
      return
    if "vehicle" in self.feature_branches:
      self._load_view_caption(ds_cfg, scenario, "vehicle", feat_dict)
    if "overhead" in self.feature_branches:
      self._load_view_caption(ds_cfg, scenario, "overhead", feat_dict)
      
  def _load_clip_features(self, path):
    return torch.from_numpy(np.load(path)).float()
  
  def _load_local(self, local_dir, view):
    frame_to_phase_path = os.path.join(local_dir, "phase.json")
    if not os.path.isfile(frame_to_phase_path):
      view["local"] = [None] * self.max_phases
      return
    with open(frame_to_phase_path, "r") as map_file:
      frame_to_phase = json.load(map_file)
    phase_to_frame = {}
    for frame, phase in frame_to_phase.items():
      if not os.path.isfile(
        os.path.join(local_dir, frame)
      ):
        continue
      if phase not in phase_to_frame:
        phase_to_frame[phase] = [frame]
      else:
        phase_to_frame[phase].append(frame)
    view["local"] = [
      self._load_clip_features(
        os.path.join(local_dir, np.random.choice(phase_to_frame[phase_idx]))
      ).flatten()
      if phase_idx in phase_to_frame else None
      for phase_idx in range(self.max_phases)
    ]
  
  def _load_features(self, idx):
    ds_cfg, scenario, in_ds_idx = self._get_scenario(idx)
    is_external = ds_cfg["bbox_vehicle"] is None
    if is_external:
      scenario = scenario.split(".")
      assert len(scenario) == 2
      scenario = scenario[0]
    
    feat_dict = {}
    self._load_caption(
      ds_cfg, 
      ds_cfg["usable"][in_ds_idx] if "usable" in ds_cfg else None, 
      scenario, 
      is_external, 
      feat_dict
    )
    
    view = feat_dict["view"]
    branch = feat_dict["branch"]
    if branch == "vehicle" and is_external:
      main_path = os.path.join(ds_cfg[self.main_feature], f"{scenario}.npy")
      if self.use_local:
        local_sub_path = scenario
      if self.sub_feature is not None:
        sub_path = os.path.join(ds_cfg[self.sub_feature], f"{scenario}.npy")
    else:
      main_dir = os.path.join(ds_cfg[self.main_feature], scenario, f"{branch}_view") # npy
      selected_video = str(sample_files(main_dir))
      main_path = os.path.join(main_dir, selected_video)
      if self.use_local:
        local_sub_path = os.path.join(scenario, f"{branch}_view", selected_video[:-4])
      if self.sub_feature is not None:
        sub_dir = os.path.join(ds_cfg[self.sub_feature], scenario, f"{branch}_view")
        sub_path = os.path.join(sub_dir, selected_video)
    
    raw_main_feats = self._load_clip_features(main_path)
    selected_frames = self._get_frames(view, len(raw_main_feats))
    main_feats = torch.index_select(raw_main_feats, 0, selected_frames)
    assert len(main_feats) == len(selected_frames)
    
    view["org_feat_len"] = len(main_feats) if len(main_feats) < self.max_feats else self.max_feats
    feat_dict["feats"] = self._resample_video_features(main_feats)
    
    if self.sub_feature is not None:
      if os.path.isfile(sub_path):
        raw_sub_feats = self._load_clip_features(sub_path)
        assert len(raw_main_feats) == len(raw_sub_feats)
        sub_feats = torch.index_select(raw_sub_feats, 0, selected_frames)
        feat_dict["sub_feats"] = self._resample_video_features(sub_feats)
      else:
        feat_dict["sub_feats"] = feat_dict["feats"]
    
    if self.use_local:
      self._load_local(os.path.join(ds_cfg["local_annotated"], local_sub_path), view)

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
    if time_token > self.num_bins:
      assert time_token + 1 <= self.num_bins
      return self.num_bins + self.num_text_tokens
    assert time_token <= self.num_bins
    return time_token + self.num_text_tokens
  
  def _get_time_token(self, x, duration, num_bins):
    time_token = int(float((num_bins - 1) * x) / float(duration))
    # if time_token > self.num_bins:
    #   assert time_token + 1 <= self.num_bins
    #   return self.num_bins
    assert time_token <= self.num_bins
    return time_token
  
  def _get_output_tokens(self, text, time_output_tokens):
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
    return torch.cat([output_tokens, torch.LongTensor([self.tokenizer.eos_token_id])], 0)
  
  def _get_output_text(self, text, time_tokens):
    time_repr = [
      "<time=" + str(st) + ">" + " " + "<time=" + str(ed) + ">"
      for st, ed in time_tokens
    ]
    assert len(time_repr) == len(text)
    return " ".join(
      [f"{time_repr[t_i]} {text[t_i]}" for t_i in range(len(text))]
    )
  
  def __getitem__(self, idx):
    if idx == 0:
      self._redistribute_cutoff_samples()
      
    feat_dict = self._load_features(idx)
    
    feat = feat_dict["feats"]
    view = feat_dict["view"]
    
    duration = view["duration"]
    phases = view["event_phase"]
    
    start = [phase["start_time"] - view["n_start"] for phase in phases]
    end = [phase["end_time"] - view["n_start"] for phase in phases]
    
    time_output_tokens = [torch.LongTensor([self.time_tokenize(st, duration, self.num_bins),
                                            self.time_tokenize(ed, duration, self.num_bins)])
                          for st, ed in zip(start, end)]
                                       
    vehicle_text = [
      self.augmentor.apply_nlp_long_sentence(p['caption_vehicle']) for p in phases
    ]                   
    vehicle_tokens = self._get_output_tokens(vehicle_text, time_output_tokens)
    pedestrian_text = [
      self.augmentor.apply_nlp_long_sentence(p['caption_pedestrian']) for p in phases
    ]  
    pedestrian_tokens = self._get_output_tokens(pedestrian_text, time_output_tokens)
    
    if self.denoising and not self.return_raw_text:
      num_phases = len(view["label_order"])
      tb_mask = [
        np.random.uniform() < self.phase_noise_density for _ in range(num_phases)
      ]
      num_mask = sum(tb_mask)
      if num_mask == 0 or num_mask == num_phases:
        rv_idx = np.random.randint(num_phases)
        tb_mask[rv_idx] = not tb_mask[rv_idx]
      denoise_vehicle_text = phase_sentinel_text_mask(vehicle_text, tb_mask)
      denoise_vehicle_tokens = self._get_output_tokens(
        denoise_vehicle_text, time_output_tokens
      )
      denoise_pedestrian_text = phase_sentinel_text_mask(pedestrian_text, tb_mask)
      denoise_pedestrian_tokens = self._get_output_tokens(
        denoise_pedestrian_text, time_output_tokens
      )
      
      vid_mask = ~np.array(tb_mask)
      start_frames = [
        self._get_time_token(st, duration, view["org_feat_len"]) for st in start
      ]
      denoising_feat = phase_sentinel_vid_mask(feat, vid_mask, start_frames)
    
    ret = {
      "feat": feat,
      "vehicle_tokens": vehicle_tokens,
      "pedestrian_tokens": pedestrian_tokens
    }
    
    if self.use_local:
      ret["local"] = view["local"]
      
    if self.sub_feature is not None:
      ret["sub_feat"] = feat_dict["sub_feats"]
    
    if not self.return_raw_text:
      if self.denoising:
        ret["denoising_feat"] = denoising_feat
        ret["denoise_vehicle_tokens"] = denoise_vehicle_tokens
        ret["denoise_pedestrian_tokens"] = denoise_pedestrian_tokens
      return ret
    
    time_tokens = [[self._get_time_token(st, duration, self.num_bins),
                    self._get_time_token(ed, duration, self.num_bins)]
                          for st, ed in zip(start, end)]
    vehicle_output_text = self._get_output_text(vehicle_text, time_tokens)
    pedestrian_output_text = self._get_output_text(pedestrian_text, time_tokens)
    
    ret["vehicle_text"] = vehicle_output_text
    ret["pedestrian_text"] = pedestrian_output_text
    
    return ret
  
  def __len__(self):
    return len(self.ds_indices)
  
  def _get_frames(self, view, total_samples):
    total_samples -= 1
    phases = view["event_phase"]
    view["label_order"] = [p["labels"][0] for p in phases]
    
    for phase in phases:
      phase["start_time"] = float(phase["start_time"])
      phase["end_time"] = float(phase["end_time"])
    
    phases = self._sort_phases(phases)
    
    max_pad_frames = self.max_pad_time * self.extract_fps
    
    true_start = phases[0]["start_time"]
    true_start_frame = int(true_start * self.extract_fps)
    true_end = phases[-1]["end_time"]
    for phase in phases:
      if phase["end_time"] > true_end:
        true_end = phase["end_time"]
    true_end_frame = int(true_end * self.extract_fps)
    
    # assert true_end_frame < total_samples or \
    #   true_end_frame - total_samples < self.extract_fps * 5
    true_end_frame = min(true_end_frame, total_samples)
    true_end = true_end_frame / self.extract_fps
    
    n_start_frame = np.random.randint(
      max(true_start_frame - max_pad_frames, 0),
      true_start_frame + 1,
    ) if self.random_pad_time else true_start_frame
    n_start = n_start_frame / self.extract_fps
    n_end_frame = np.random.randint(
      true_end_frame,
      min(true_end_frame + max_pad_frames, total_samples) + 1,
    ) if self.random_pad_time else true_end_frame
    n_end = n_end_frame / self.extract_fps
    
    assert phases[0]["start_time"] >= n_start and \
      (true_end <= n_end or true_end - n_end < 1)
    n_end = max(n_end, true_end)
    n_end_frame = int(n_end * self.extract_fps)
      
    sampled_frames = sample_every(
      n_end_frame - n_start_frame,
      self.extract_fps // self.sample_fps,
      n_start_frame
    )
    assert sampled_frames[-1] <= n_end_frame
    
    for phase in phases:
      if phase["start_time"] > n_end:
        phase["start_time"] = n_end
        phase["end_time"] = n_end
      elif phase["end_time"] > n_end:
        phase["end_time"] = n_end
    
    view["event_phase"] = phases
    view["n_start"] = n_start
    view["n_end"] = n_end
    view["duration"] = n_end - n_start
    
    return torch.tensor(sampled_frames)
  
  def _sort_phases(self, phases):
    sorted_phases = sorted(phases, key=lambda p: p["start_time"])
    # for idx in range(len(phases) - 1):
      # if sorted_phases[idx]["end_time"] > sorted_phases[idx + 1]["end_time"]:
      #   print([{"st": p["start_time"], "ed": p["end_time"]} for p in sorted_phases])
      # assert sorted_phases[idx]["end_time"] <= sorted_phases[idx + 1]["end_time"]
    return sorted_phases
  

# print([{"st": p["start_time"], "ed": p["end_time"]} for p in sorted_phases])
