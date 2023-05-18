import shutil
from ..logger import logging

def save_copy(src, dst):
    # type: (str, str) -> None
    logging.info('Saving copy from {} to {}'.format(src, dst))
    shutil.copy2(src, dst) # Copy2 preserves metadata and accepts a dst directory