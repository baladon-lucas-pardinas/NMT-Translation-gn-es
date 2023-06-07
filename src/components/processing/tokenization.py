import re
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Done to avoid GPU warnings from Tensorflow
from src.components.processing.cleaning import clean_text, clean_token

import spacy

def get_tokenizer():
    # type: () -> spacy.lang.es.Spanish
    tokenizer = None
    current_dir = os.path.dirname(os.path.realpath(__file__))
    nlp_path = os.path.join(current_dir, "spacy_nlp.p")

    if os.path.isfile(nlp_path):
        with open(nlp_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = spacy.load("es_core_news_md")
        pickle.dump(tokenizer, open(nlp_path, "wb"))

    return tokenizer

def tokenize(tokenizer, text):
    # type: (str, spacy.lang.es.Spanish) -> list
    tokens = []
    #text = clean_text(text)

    doc = tokenizer(text)
    for token in doc:
        cleaned_token = clean_token(token.text)
        if cleaned_token != '':
            tokens.append(cleaned_token)
        else:
            text = clean_text(text)
            tokens = [token.strip() for token in text.split()]

    return tokens