import warnings
import logging
from functools import wraps

def warning_filter(logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger == 'spacy':
                warning_filter_spacy(func, *args, **kwargs)
            else:
                warning_filter_default(func, *args, **kwargs)
        return wrapper
    return decorator

def warning_filter_spacy(func, *args, **kwargs):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Needs to be done before importing spacy
    logging.getLogger('spacy').setLevel(logging.ERROR)
    func(*args, **kwargs)
    logging.getLogger('spacy').setLevel(logging.WARNING)

def warning_filter_default(func, *args, **kwargs):
    warnings.simplefilter('ignore')
    func(*args, **kwargs)
    warnings.simplefilter('default')