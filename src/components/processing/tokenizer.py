import re
import pickle
import os
__spacy_module = None
__nlp = None

MAIL_TOKEN = '<MAIL>'
URL_TOKEN = '<URL>'
NUMBER_TOKEN = '<NUMBER>'
FLOAT_NUMBER = NUMBER_TOKEN
IP_TOKEN = NUMBER_TOKEN

# Normalize raw text
# Normalize "pusó" symbols (https://es.wikipedia.org/wiki/Alfabeto_guaran%C3%AD) 
def clean_text(text):
    # type: (str) -> str
    clean_text = re.sub("’|`|^", "'", text) # TODO: Hablar con Santiago (agregué | |)
    clean_text = re.sub("´`|´´", "'", clean_text)
    clean_text = re.sub("’|`|ʼ", "'", clean_text)

    clean_text = clean_text.lower()

    #Reduce vocabulary. TODO: Hablar con Santiago
    clean_text = re.sub("(\w+)@(\w+)\.(\w+)", '', clean_text)
    clean_text = re.sub("http\S+", '', clean_text)
    clean_text = re.sub("[+-]?([0-9]*[.])?[0-9]+", '', clean_text)
    clean_text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', clean_text)

    return clean_text

# Convert non-characters to blank space
def clean_token(token):
    # type: (str) -> str
    token = re.sub(' ','',token)
    token = re.sub('\n|\t|\.|,|%|-','',token)
    return token
    
def check_tokenizer_module():
    # type: () -> bool
    global __spacy_module
    if __spacy_module is None:
        try:
            import spacy
            __spacy_module = spacy
            global __nlp

            # Cache doc
            current_dir = os.path.dirname(os.path.realpath(__file__))
            nlp_path = os.path.join(current_dir, "spacy_nlp.p")
            if os.path.isfile(nlp_path):
                with open(nlp_path, "rb") as f:
                    __nlp = pickle.load(f)
            else:
                __nlp = spacy.load("es_core_news_md")
                pickle.dump(__nlp, open(nlp_path, "wb"))
        except ImportError:
            return False
    return True

def tokenize(text):
    # type: (str) -> list
    tokens = []
    text = clean_text(text)

    global __spacy_module
    if __spacy_module is not None:
        global __nlp
        doc = __nlp(text)
        for token in doc:
            cleaned_token = clean_token(token.text)
            if cleaned_token != '':
                tokens.append(cleaned_token)
    else:
        text = clean_text(text)
        tokens = [token for token in text.split()]

    return tokens