import torch
import torch.nn as nn


def freeze_module(module: nn.Module):
  for param in module.parameters():
    param.requires_grad = False
    

def total_parameters(module: nn.Module):
  return sum(p.numel() for p in module.parameters() if p.requires_grad)


def clone_and_subsample_pos_embed(pos_embed: torch.Tensor, target_features: int):
  sub_pos = []
  for j in range(target_features):
    sub_pos.append(
      pos_embed[:, (j * pos_embed.shape[1]) // target_features].detach().clone()
    )
  return torch.cat(sub_pos) # (target_features, pos_embed.shape[2])
