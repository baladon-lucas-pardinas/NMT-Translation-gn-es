from model import model
from ..utils import command_handler

def train(model_name, config, save_each_epochs=None):
    # type: (str, command_handler.CommandConfig, int) -> None
    model.train(model_name, config, save_each_epochs)