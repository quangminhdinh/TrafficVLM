import numpy as np
from BackTranslation import BackTranslation


class BackTransAug:
  
  def __init__(self, src='en', tmp = 'zh-cn') -> None:
    self.aug = BackTranslation()
    self.src = src
    self.tmp = tmp
    
  def run(self, text):
    result = self.aug.translate(text, src=self.src, tmp = self.tmp)
    return result.result_text
  
  def __repr__(self):
    return self.__class__.__name__


class Augmentor:
  
  def __init__(self, cfg, nlp_augs=[]) -> None:
    self.ALL_NLP_AUGS = ["backtrans"]

    self.nlp_augs = []
    self.nlp_prob = cfg.NLP_PROB
    for nlp_aug_name in nlp_augs:
      self._add_nlp_aug(nlp_aug_name)
    print(f"Using {len(self.nlp_augs)} nlp augmentors in total: "
          f"{', '.join(self.nlp_augs.__class__.__name__)}.")
    
  def _add_nlp_aug(self, name):
    if name not in self.ALL_NLP_AUGS:
      print(f"The is no augmentation name {name}," 
            f"please select from {self.ALL_NLP_AUGS}!")
    if name == "backtrans":
      aug = BackTransAug()
      self.nlp_augs.append(aug)
      
  def _apply_nlp_single(self, text):
    if np.random.uniform > self.nlp_prob:
      return text
    aug = np.random.choice(self.nlp_augs)
    return aug.run(text)
      
  def apply_nlp_long_sentence(self, text: str):
    if len(self.nlp_augs) == 0:
      return simple_text_preprocess(text)
    processed = self.apply_nlp_multi(text.split('.'))
    processed = [simple_text_preprocess(txt) for txt in processed]
    return " ".join(processed)
      
  def apply_nlp_multi(self, text_list):
    if len(self.nlp_augs) == 0:
      return text_list
    return [
      self._apply_nlp_single(text) for text in text_list
    ]


def simple_text_preprocess(text: str):
  text = text.strip()
  text = text.lower()
  if text[-1] != '.':
    text = text + '.'
  return text
