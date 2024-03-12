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
      
  def forward(self, feats, target_embed, output_tokenized): # (feats, proj)
    feats = self.visual_encoder(feats) # B T D
    if self.proj_v2t is not None:
      feats = self.proj_v2t(feats)
    feats_atts = torch.ones(feats.size()[:-1], dtype=torch.long).to(feats.device)
    encoded = BaseModelOutput(
      last_hidden_state=torch.cat([feats, target_embed], dim=1) # type: ignore
    )
    tgt_embed_atts = torch.ones(target_embed.size()[:-1], dtype=torch.long).to(feats.device)
    encoder_atts = torch.cat([feats_atts, tgt_embed_atts], dim=1)
    
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
    feats,
    target_embed,
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
      target_embed (torch.Tensor): A tensor of shape (batch_size, T, D).
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
    feats = self.visual_encoder(feats) # B T D
    if self.proj_v2t is not None:
      feats = self.proj_v2t(feats)
    feats_atts = torch.ones(feats.size()[:-1], dtype=torch.long).to(feats.device)
    encoded = BaseModelOutput(
      last_hidden_state=torch.cat([feats, target_embed], dim=1) # type: ignore
    )
    tgt_embed_atts = torch.ones(target_embed.size()[:-1], dtype=torch.long).to(feats.device)
    encoder_atts = torch.cat([feats_atts, tgt_embed_atts], dim=1)
    
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
