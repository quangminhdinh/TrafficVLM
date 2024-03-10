def simple_text_preprocess(text: str):
  text = text.strip()
  text = text.lower()
  if text[-1] != '.':
    text = text + '.'
  return text
