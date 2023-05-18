import os
from ..logger import logging

def run_command(command):
    # type: (str) -> None
    logging.info('Running command: {}'.format(command))
    os.system(command)
    logging.info('Command finished: {}'.format(command))