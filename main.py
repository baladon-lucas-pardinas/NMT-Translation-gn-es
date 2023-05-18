import sys
from src.model import model
from src.utils import command_handler
from src.logger import logging
from src import config

if __name__ == '__main__':
    command_path = sys.argv[1]
    save_each_epochs = int(sys.argv[2])
    flags = sys.argv[3:]
    
    flags = ' '.join(flags)
    flags = command_handler.parse_flags(flags)
    command_config = config.get_command_config(command_path=command_path, flags=flags)
    logging.info('Training model with config {}'.format(command_config))

    try:
        model.train(command_config.command_name, command_config, save_each_epochs=save_each_epochs)
    except Exception as e:
        logging.error('Error while training with config {}'.format(command_config))
        raise e

    