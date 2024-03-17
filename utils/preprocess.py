import numpy as np
import nlpaug.augmenter.word as naw


class Augmentor:
  
  def __init__(self, cfg, nlp_augs=[], batch_size = 32) -> None:
    self.ALL_NLP_AUGS = ["backtrans"]

    self.nlp_augs = []
    self.nlp_prob = cfg.NLP_PROB
    self.nlp_per_phase_prob = cfg.NLP_PER_PHASE_PROB
    self.device = cfg.DEVICE
    self.batch_size = batch_size
    
    for nlp_aug_name in nlp_augs:
      self._add_nlp_aug(nlp_aug_name)
    print(f"Using {len(self.nlp_augs)} nlp augmentors in total: "
          f"{', '.join([aug.__class__.__name__ for aug in self.nlp_augs])}.")
    
  def _add_nlp_aug(self, name):
    if name not in self.ALL_NLP_AUGS:
      print(f"The is no augmentation name {name}," 
            f"please select from {self.ALL_NLP_AUGS}!")
    if name == "backtrans":
      if self.device.lower() != "cpu":
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
      aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en',
        device=self.device,
        batch_size=self.batch_size
      )
      self.nlp_augs.append(aug)
      
  def _apply_nlp_single(self, text):
    if np.random.uniform() > self.nlp_prob:
      return text
    aug = np.random.choice(self.nlp_augs)
    return aug.augment(text)
      
  def apply_nlp_long_sentence(self, text: str):
    if len(self.nlp_augs) == 0 or np.random.uniform() > self.nlp_prob:
      return simple_text_preprocess(text)
    processed = self.apply_nlp_multi(text.split('.'))
    processed = [simple_text_preprocess(txt) if len(txt) > 0 else txt for txt in processed]
    return " ".join(processed)
      
  def apply_nlp_multi(self, text_list):
    if len(self.nlp_augs) == 0:
      return text_list
    tb_augmented = [np.random.uniform() < self.nlp_per_phase_prob 
                    for _ in range(len(text_list))]
    aug_list = [text for text, aug in zip(text_list, tb_augmented) if aug]
    if len(aug_list) == 0:
      return text_list
    aug = np.random.choice(self.nlp_augs)
    aug_list = aug.augment(aug_list)
    aug_idx = 0
    ret = []
    for org_text, aug in zip(text_list, tb_augmented):
      if aug:
        ret.append(aug_list[aug_idx])
        aug_idx += 1
      else:
        ret.append(org_text)
    return ret


def simple_text_preprocess(text: str):
  text = text.strip()
  text = text.lower()
  if text[-1] != '.':
    text = text + '.'
  return text
