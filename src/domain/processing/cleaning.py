# -*- coding: utf-8 -*-
import re

DATE_REGEX   = r"\d{1,2}\/\d{1,2}\/\d{2,4}"
MAIL_REGEX   = r"(\w+)@(\w+)\.(\w+)"
URL_REGEX    = r"http\S+"
NUMBER_REGEX = r"[+-]?([0-9]*[.])?[0-9]+"
IP_REGEX     = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"

# Normalize raw text
# Normalize "pusó" symbols (https://es.wikipedia.org/wiki/Alfabeto_guaran%C3%AD) 
def normalize_text(text, lowercase=False):
    # type: (str, bool) -> str
    cleaned_text = text

    if lowercase:
        cleaned_text = cleaned_text.lower()

    cleaned_text = re.sub("’`^", "'", cleaned_text)
    cleaned_text = re.sub("´`|´´", "'", cleaned_text)
    cleaned_text = re.sub("’|`|ʼ", "'", cleaned_text)

    return cleaned_text

def reduce_vocabulary(text):
    # type: (str) -> str
    cleaned_text = text
    cleaned_text = re.sub(DATE_REGEX,   '', cleaned_text)
    cleaned_text = re.sub(MAIL_REGEX,   '', cleaned_text)
    cleaned_text = re.sub(URL_REGEX,    '', cleaned_text)
    cleaned_text = re.sub(NUMBER_REGEX, '', cleaned_text)
    cleaned_text = re.sub(IP_REGEX,     '', cleaned_text)
    return cleaned_text

def clean_text(text, reduce_vocab=True, normalize=False):
    # type: (str, bool, bool) -> str
    cleaned_text = text

    if normalize:
        cleaned_text = normalize_text(cleaned_text)
    
    if reduce_vocab:
        cleaned_text = reduce_vocabulary(cleaned_text)
        
    return cleaned_text

# Convert non-characters to blank spaces
def clean_token(token, lowercase=False, clean_punctuation=True, clean_spaces=True):
    # type: (str, bool, bool, bool) -> str
    if lowercase:
        token = token.lower()

    if clean_punctuation:
        token = re.sub('\.|,|%|-|\{|\}|\(|\)','', token) # strings.punctuation could be used too

    if clean_spaces:
        token = re.sub('\n|\t|[ ]','', token)
    return token