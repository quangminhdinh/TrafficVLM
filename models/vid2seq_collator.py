import torch
import torch.nn as nn
import torch.nn.functional as F

from .vid2seq import Vid2Seq
from .utils import freeze_module, total_parameters
from .modeling_t5 import T5ForConditionalGeneration
from .mlp import Mlp


class Vid2SeqCollator(nn.Module):
  
  def __init__(self, cfg,
               tokenizer,
               num_bins=100,
               num_features=100,
               features_dim=768) -> None:
    
    super().__init__()
    
    self.num_bins = num_bins
    self.label_smoothing = cfg.LABEL_SMOOTHING
    
    print("\nInitializing Vid2Seq model...")
    self.model = Vid2Seq(cfg, tokenizer, num_bins, num_features)
    assert type(self.model.t5_model) is T5ForConditionalGeneration
    pre_params = total_parameters(self.model)
    freeze_module(self.model.t5_model.encoder)
    print("Number of freezed encoder parameters:", pre_params - total_parameters(self.model))
    
    self.pretrained_ckpt = cfg.VID2SEQ_PATH
    self.load_ckpt = cfg.LOAD_VID2SEQ_CKPT
    if self.load_ckpt:
      assert self.pretrained_ckpt is not None
    
    self.vehicle_embed = nn.Parameter(torch.rand(cfg.TARGET_EMBED_SIZE, features_dim))
    self.pedestrian_embed = nn.Parameter(torch.rand(cfg.TARGET_EMBED_SIZE, features_dim))
      
  def load_pretrained(self, model_ckpt):
    self.model.load_state_dict(model_ckpt, strict=False)
    
  def _fill_overhead(self, overhead_feats):
    overhead_feats = [feat 
                      if feat is not None 
                      else self.overhead_unk 
                      for feat in overhead_feats]
    return torch.stack(overhead_feats) # type: ignore
  
  def calculate_loss_from_logits(self, logits, labels):
    return F.cross_entropy(
      logits.view(-1, logits.size(-1)), 
      labels.view(-1), 
      ignore_index=-100, 
      label_smoothing=self.label_smoothing
    )
  
  @torch.no_grad()
  def normalize_time_embeddings(self):
    t5 = self.model.t5_model
    assert isinstance(t5, T5ForConditionalGeneration)
    frozen_norm = torch.norm(t5.shared.weight[:-self.num_bins, :], dim=1).mean(0)
    trainable_weight = t5.shared.weight[-self.num_bins:, :]
    t5.shared.weight[-self.num_bins:, :].div_(
      torch.norm(trainable_weight, dim=1).mean(0) / frozen_norm
    )

    frozen_norm = torch.norm(t5.lm_head.weight[:-self.num_bins, :], dim=1).mean(0)
    trainable_weight = t5.lm_head.weight[-self.num_bins:, :]
    t5.lm_head.weight[-self.num_bins:, :].div_(
      torch.norm(trainable_weight, dim=1).mean(0) / frozen_norm
    )
  
  def forward(self, feats, output_tokens, tgt_type):
    if tgt_type == "vehicle":
      tgt_embed = self.vehicle_embed
    elif tgt_type == "pedestrian":
      tgt_embed = self.pedestrian_embed
    tgt_embed = torch.unsqueeze(tgt_embed, 0)
    tgt_embed = tgt_embed.repeat(len(feats), 1, 1)
    return self.model(
      feats,
      tgt_embed,
      {'input_ids': output_tokens, 'attention_mask': output_tokens != 0}
    )
    
  @torch.no_grad()
  def generate(
    self,
    feats,
    tgt_type,
    use_nucleus_sampling=False,
    num_beams=4,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_captions=1,
    temperature=1,
  ):
    """
    Args:
      feats (torch.Tensor): A tensor of shape (batch_size, T, D).
      tgt_type (string): Output type (vehicle or pedestrian).
      use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
      num_beams (int): Number of beams for beam search. 1 means no beam search.
      max_length (int): The maximum length of the sequence to be generated.
      min_length (int): The minimum length of the sequence to be generated.
      top_p (float): The cumulative probability for nucleus sampling.
      repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
      num_captions (int): Number of captions to be generated for each image.
    Returns:
      captions (list): A list of strings of length batch_size * num_captions.
    """
    if tgt_type == "vehicle":
      tgt_embed = self.vehicle_embed
    elif tgt_type == "pedestrian":
      tgt_embed = self.pedestrian_embed
    tgt_embed = torch.unsqueeze(tgt_embed, 0)
    tgt_embed = tgt_embed.repeat(len(feats), 1, 1)
    return self.model.generate(
      feats,
      tgt_embed,
      use_nucleus_sampling=use_nucleus_sampling,
      num_beams=num_beams,
      max_length=max_length,
      min_length=min_length,
      top_p=top_p,
      repetition_penalty=repetition_penalty,
      length_penalty=length_penalty,
      num_captions=num_captions,
      temperature=temperature,
    )
