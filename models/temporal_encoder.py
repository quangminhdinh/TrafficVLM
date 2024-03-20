import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clone_and_subsample_pos_embed


class DependentTemporalEncoder(nn.Module):
  
  def __init__(self, vit_model, num_features=6) -> None:
    super().__init__()
    
    ## Positional Embeddings
    self.pos_embed = nn.Parameter(
      clone_and_subsample_pos_embed(vit_model.pos_embed, num_features).unsqueeze(0)
    )
    self.pos_drop = vit_model.pos_drop

    ## Attention Blocks
    self.blocks = vit_model.blocks
    self.norm = vit_model.norm

  @torch.jit.ignore # type: ignore
  def no_weight_decay(self):
    return {'pos_embed'}

  def forward(self, x):
    ## resizing the positional embeddings in case they don't match the input at inference
    if x.size(1) != self.pos_embed.size(1):
      time_embed = self.pos_embed.transpose(1, 2)
      new_time_embed = F.interpolate(time_embed, size=(x.size(1)), mode='nearest')
      new_time_embed = new_time_embed.transpose(1, 2)
      x = x + new_time_embed
    else:
      x = x + self.pos_embed
    x = self.pos_drop(x)

    ## Attention blocks
    for blk in self.blocks:
      x = blk(x)

    x = self.norm(x)
    return x

