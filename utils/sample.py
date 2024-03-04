import os
import numpy as np


def sample_bins(length, bins=1):
  samples_per_bin = length // bins
  ids = []
  for _bin in range(bins):
    num_samples = samples_per_bin if _bin < bins - 1 else samples_per_bin + length % bins
    ids.append(_bin * samples_per_bin + np.random.randint(num_samples))
  return ids


def sample_files(parent_path, num_samples=1, ret_ids=False):
  all_samples = os.listdir(parent_path)
  total_samples = len(all_samples)
  
  if num_samples == 1:
    return np.random.randint(total_samples) if ret_ids else np.random.choice(all_samples)
  ids = sample_bins(total_samples, num_samples)
  return ids if ret_ids else [all_samples[idx] for idx in ids]



