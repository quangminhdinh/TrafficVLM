import torch

from .wts_solver import WTSSolver
from .logging import setup_logging


def get_solver(*args, **kwargs):
  return WTSSolver(*args, **kwargs)


def get_optimizer(cfg, params):
  return torch.optim.Adam(
    params,
    cfg.LR,
    betas=(cfg.BETA1, cfg.BETA2),
    weight_decay=cfg.WEIGHT_DECAY,
  )
