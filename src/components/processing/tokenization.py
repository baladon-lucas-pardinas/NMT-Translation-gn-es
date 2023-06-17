import re
import pickle
import os
from abc import ABC, abstractmethod

from src.components.processing.cleaning import clean_text, clean_token

class Tokenizer(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def tokenize(self, text):
        ...

class SpacyTokenizer(Tokenizer):
    def __init__(self):
        import spacy
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Done to avoid GPU warnings from Tensorflow
        tokenizer = None
        current_dir = os.path.dirname(os.path.realpath(__file__))
        nlp_path = os.path.join(current_dir, "spacy_nlp.p")

        if os.path.isfile(nlp_path):
            with open(nlp_path, "rb") as f:
                tokenizer = pickle.load(f)
        else:
            tokenizer = spacy.load("es_core_news_md")
        pickle.dump(tokenizer, open(nlp_path, "wb"))
        super().__init__(tokenizer)

    def tokenize(self, text):
        # type: (str) -> list
        tokens = []
        #text = clean_text(text)

        doc = self.tokenizer(text)
        for token in doc:
            cleaned_token = clean_token(token.text)
            if cleaned_token != '':
                tokens.append(cleaned_token)
            else:
                text = clean_text(text)
                tokens = [token.strip() for token in text.split()]

        return tokens

class NLTKTokenizer(Tokenizer):
    def __init__(self):
        import nltk
        tokenizer = nltk.tokenize.word_tokenize
        super().__init__(tokenizer)

    def tokenize(self, text):
        # type: (str) -> list
        tokens = self.tokenizer(text)
        tokens = [clean_token(token) for token in tokens]
        tokens = [token for token in tokens if token != '']
        return tokens

def get_tokenizer(tokenizer='nltk'):
    # type: (str) -> Tokenizer
    tokenizer = tokenizer.lower()
    if tokenizer == 'nltk':
        return NLTKTokenizer()
    elif tokenizer == 'spacy':
        return SpacyTokenizer()
    else:
        raise ValueError('Tokenizer {} not found'.format(tokenizer))
