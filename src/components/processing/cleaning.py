# -*- coding: utf-8 -*-
import re

MAIL_REGEX   = r"(\w+)@(\w+)\.(\w+)"
URL_REGEX    = r"http\S+"
NUMBER_REGEX = r"[+-]?([0-9]*[.])?[0-9]+"
IP_REGEX     = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"

# Normalize raw text
# Normalize "pusó" symbols (https://es.wikipedia.org/wiki/Alfabeto_guaran%C3%AD) 
def clean_text(text):
    # type: (str) -> str
    clean_text = re.sub("’|`|^", "'", text) # TODO: Hablar con Santiago (agregué | |)
    clean_text = re.sub("´`|´´", "'", clean_text)
    clean_text = re.sub("’|`|ʼ", "'", clean_text)

    clean_text = clean_text.lower()

    #Reduce vocabulary. TODO: Hablar con Santiago
    clean_text = re.sub(MAIL_REGEX,   '', clean_text)
    clean_text = re.sub(URL_REGEX,    '', clean_text)
    clean_text = re.sub(NUMBER_REGEX, '', clean_text)
    clean_text = re.sub(IP_REGEX,     '', clean_text)

    return clean_text

# Convert non-characters to blank space
def clean_token(token):
    # type: (str) -> str
    token = token.strip()
    #token = re.sub(' ','',token)
    #token = re.sub('\n|\t|\.|,|%|-','',token)
    return token