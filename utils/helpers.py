import os
import numpy as np


def load_all_feats_from_disk(paths_list):
  feats = [np.load(path) for path in paths_list]
  return np.array(feats)


def get_all_top_dirs(parent_path):
  return [sub_dir for sub_dir in os.listdir(parent_path) \
    if os.path.isdir(os.path.join(parent_path, sub_dir))]


def get_all_top_dirs_full(parent_path):
  return [os.path.join(parent_path, sub_dir) \
    for sub_dir in get_all_top_dirs(parent_path)]
