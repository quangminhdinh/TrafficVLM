import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer


class Vid2Seq(nn.Module):
  
  def __init__(self, cfg,
               tokenizer,
               num_bins=100,
               num_features=100):
    
    super().__init__()
    
    self.t5_model = T5ForConditionalGeneration.from_pretrained(
      decoder_dropout=cfg.DEC_DROP,
      label_smoothing=cfg.LABEL_SMOOTHING,
      pretrained_model_name_or_path=cfg.T5_PATH,
      is_gated_act="v1_1" in cfg.T5_PATH,
    )
    assert type(self.t5_model) is T5ForConditionalGeneration
    
    self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)  # remove the weights of the 28 tokens that are not used (32128 vs 32100 in the tokenizer) 
    self.t5_model.resize_token_embeddings(len(tokenizer))  # add time tokens 
    
    self.visual_encoder = VisionTransformer(
      num_features=num_features,
      embed_dim=cfg.EMBED_DIM,
      depth=cfg.DEPTH,
      num_heads=cfg.NUM_HEADS,
      mlp_dim=cfg.MLP_DIM,
      qkv_bias=True,
      qk_scale=None,
      drop_rate=cfg.VIS_DROP,
      attn_drop_rate=cfg.VIS_DROP,
      norm_layer=nn.LayerNorm
    )
    self.t5_tokenizer = tokenizer
    self.feature_branches = cfg.FEATURE_BRANCHES
    self.proj_v2t = None
    if self.t5_model.model_dim != 768:
      self.proj_v2t = nn.Linear(768, self.t5_model.model_dim)
      
  def forward(self, vehicle, overhead, output_tokenized): # (feats, proj)
    vehicle_feats, vehicle_proj = vehicle
    vehicle_feats = self.visual_encoder(vehicle_feats) # B T D
    if vehicle_proj is not None:
      vehicle_feats = vehicle_proj(vehicle_feats)
    if self.proj_v2t is not None:
      vehicle_feats = self.proj_v2t(vehicle_feats)
    atts_vehicle = torch.ones(vehicle_feats.size()[:-1], 
                              dtype=torch.long).to(vehicle_feats.device)
    
    if "overhead" in self.feature_branches:
      overhead_feats, overhead_proj = overhead
      overhead_feats = self.visual_encoder(overhead_feats) # B T D
      if overhead_proj is not None:
        overhead_feats = overhead_proj(overhead_feats)
      if self.proj_v2t is not None:
        overhead_feats = self.proj_v2t(overhead_feats)
      atts_overhead = torch.ones(overhead_feats.size()[:-1], 
                                dtype=torch.long).to(overhead_feats.device)
      
      encoded = BaseModelOutput(
        last_hidden_state=torch.cat([vehicle_feats, overhead_feats], dim=1) # type: ignore
      )
      encoder_atts = torch.cat([atts_vehicle, atts_overhead], dim=1)
    else:
      encoded = BaseModelOutput(last_hidden_state=vehicle_feats)
      encoder_atts = atts_vehicle
    
    targets = output_tokenized['input_ids'].masked_fill(
      output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
    )
    
    assert type(self.t5_model) is T5ForConditionalGeneration
    outputs = self.t5_model(
      encoder_outputs=encoded,
      attention_mask=encoder_atts,
      decoder_attention_mask=output_tokenized['attention_mask'],
      return_dict=True,
      labels=targets,
    )
    loss = outputs.loss
    
    return loss
  
  @torch.no_grad()
  def generate(
    self,
    vehicle,
    overhead,
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
      vehicle (torch.Tensor, nn.Module | None): A tensor of shape (batch_size, T, D) and an additional projection layer.
      overhead (torch.Tensor, nn.Module | None): A tensor of shape (batch_size, T, D) and an additional projection layer.
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
    vehicle_feats, vehicle_proj = vehicle
    vehicle_feats = self.visual_encoder(vehicle_feats) # B T D
    if vehicle_proj is not None:
      vehicle_feats = vehicle_proj(vehicle_feats)
    if self.proj_v2t is not None:
      vehicle_feats = self.proj_v2t(vehicle_feats)
    atts_vehicle = torch.ones(vehicle_feats.size()[:-1], 
                              dtype=torch.long).to(vehicle_feats.device)
    
    if "overhead" in self.feature_branches:
      overhead_feats, overhead_proj = overhead
      overhead_feats = self.visual_encoder(overhead_feats) # B T D
      if overhead_proj is not None:
        overhead_feats = overhead_proj(overhead_feats)
      if self.proj_v2t is not None:
        overhead_feats = self.proj_v2t(overhead_feats)
      atts_overhead = torch.ones(overhead_feats.size()[:-1], 
                                dtype=torch.long).to(overhead_feats.device)
      
      encoded = BaseModelOutput(
        last_hidden_state=torch.cat([vehicle_feats, overhead_feats], dim=1) # type: ignore
      )
      encoder_atts = torch.cat([atts_vehicle, atts_overhead], dim=1)
    else:
      encoded = BaseModelOutput(last_hidden_state=vehicle_feats)
      encoder_atts = atts_vehicle
    
    assert type(self.t5_model) is T5ForConditionalGeneration
    outputs = self.t5_model.generate(
      encoder_outputs=encoded,
      attention_mask=encoder_atts,
      do_sample=use_nucleus_sampling,
      top_p=top_p,
      temperature=temperature,
      num_beams=num_beams,
      max_new_tokens=max_length,
      min_length=min_length,
      repetition_penalty=repetition_penalty,
      length_penalty=length_penalty,
      num_return_sequences=num_captions,
    )
    output_text = self.t5_tokenizer.batch_decode(
      outputs, skip_special_tokens=True
    )
    
    return output_text


import torch
import torch.nn as nn

from .utils import freeze_module, total_parameters
from .modeling_t5 import T5ForConditionalGeneration
from .mlp import Mlp


class TrafficVLM(nn.Module):
  
  def __init__(self, cfg,
               tokenizer,
               num_bins=100,
               num_features=100,
               features_dim=768) -> None:
    
    super().__init__()
    
    self.num_bins = num_bins
    
    print("\nInitializing Vid2Seq model...")
    self.model = Vid2Seq(cfg, tokenizer, num_bins, num_features)
    assert type(self.model.t5_model) is T5ForConditionalGeneration
    pre_params = total_parameters(self.model)
    freeze_module(self.model.t5_model.encoder)
    print("Number of freezed encoder parameters:", total_parameters(self.model) - pre_params)
    
    self.use_vehicle_proj = cfg.VEHICLE_PROJ
    self.use_overhead_proj = cfg.OVERHEAD_PROJ and "overhead" in cfg.FEATURE_BRANCHES
    
    self.vehicle_proj = Mlp(
      self.model.t5_model.model_dim, 
      cfg.FEAT_MLP_DIM
    ) if self.use_vehicle_proj else None
    self.overhead_proj = Mlp(
      self.model.t5_model.model_dim, 
      cfg.FEAT_MLP_DIM
    ) if self.use_overhead_proj else None
    
    if "overhead" in self.model.feature_branches:
      self.overhead_unk = nn.Parameter(torch.zeros(features_dim))
    
    self.pretrained_ckpt = cfg.VID2SEQ_PATH
    self.load_ckpt = cfg.LOAD_VID2SEQ_CKPT
    if self.load_ckpt:
      assert self.pretrained_ckpt is not None
      
  def load_pretrained(self, model_ckpt):
    self.model.load_state_dict(model_ckpt, strict=False)
    
  def _fill_overhead(self, overhead_feats):
    overhead_feats = [feat 
                      if feat is not None 
                      else self.overhead_unk 
                      for feat in overhead_feats]
    return torch.stack(overhead_feats) # type: ignore
  
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
  
  def forward(self, vehicle_feats, overhead_feats, output_tokens):
    overhead_feats = self._fill_overhead(overhead_feats)
    assert overhead_feats.size() == vehicle_feats.size()
    
    return self.model(
      (vehicle_feats, self.vehicle_proj),
      (overhead_feats, self.overhead_proj),
      {'input_ids': output_tokens, 'attention_mask': output_tokens != 0}
    )
    
  @torch.no_grad()
  def generate(
    self,
    vehicle,
    overhead,
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
      vehicle (torch.Tensor): A tensor of shape (batch_size, T, D).
      overhead (torch.Tensor): A tensor of shape (batch_size, T, D).
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
    overhead = self._fill_overhead(overhead)
    assert overhead.size() == vehicle.size()
    
    return self.model.generate(
      (vehicle, self.vehicle_proj),
      (overhead, self.overhead_proj),
      use_nucleus_sampling,
      num_beams,
      max_length,
      min_length,
      top_p,
      repetition_penalty,
      length_penalty,
      num_captions,
      temperature,
    )

