import torch.nn as nn


def freeze_module(module: nn.Module):
  for param in module.parameters():
    param.requires_grad = False
    

def total_parameters(module: nn.Module):
  return sum(p.numel() for p in module.parameters() if p.requires_grad)
