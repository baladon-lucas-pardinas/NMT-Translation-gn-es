import logging
import os
from datetime import datetime

CURRENT_DIR = os.getcwd()
LOG_FILE = '{}.log'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
LOG_PATH = os.path.join(CURRENT_DIR, 'logs', 'project')

if not os.path.exists(os.path.dirname(LOG_PATH)):
    os.makedirs(os.path.dirname(LOG_PATH))

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)