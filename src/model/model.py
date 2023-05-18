from . import marian
from ..utils import command_handler

def train(model, config, save_each_epochs=None):
    # type: (str, command_handler.CommandConfig, int) -> None
    train_functions = {
        'marian': marian.train,
    }
    train_function = train_functions.get(model, None)
    if train_function is None:
        raise KeyError('Model {} not found'.format(model))
    train_function(config, save_each_epochs)
