import warnings

def warning_filter(func):
    def wrapper(*args, **kwargs):
        warnings.simplefilter('ignore')
        func(*args, **kwargs)
        warnings.simplefilter('default')
    return wrapper