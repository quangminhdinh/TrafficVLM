def simple_text_preprocess(text: str):
  text = text.strip()
  text = text.capitalize()
  if text[-1] != '.':
    text = text + '.'
  return text
