import re

def __clean_text(text):
    # type: (str) -> str
    text = text.replace('\n', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text