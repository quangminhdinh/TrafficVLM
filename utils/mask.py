import torch


def phase_sentinel_text_mask(text_list, mask_list):
  count = 0
  masked_text = []
  for text, mask in zip(text_list, mask_list):
    if mask:
      masked_text.append(f"<extra_id_{count}>")
      count += 1
    else:
      masked_text.append(text)
  assert len(masked_text) == len(text_list)
  return masked_text


def phase_sentinel_vid_mask(feat, mask_list, start_frames):
  denoising_feat = feat.detach().clone()
  for idx, (start, mask) in enumerate(zip(start_frames, mask_list)):
    if not mask:
      continue
    if idx == len(mask_list) - 1:
      denoising_feat[start:] = torch.zeros(
        len(denoising_feat[start:]), denoising_feat.shape[1]
      )
      continue
    end = start_frames[idx + 1]
    denoising_feat[start : end] = torch.zeros(
      len(denoising_feat[start : end]), denoising_feat.shape[1]
    )
  return denoising_feat
