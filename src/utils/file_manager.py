import os
import shutil
from ..logger import logging

def move_files(src, dst, copy=True):
    # type: (str, str, bool) -> None
    files_in_src = os.listdir(src)
    for src_filename in files_in_src:
        src_old_file = os.path.join(src, src_filename)
        dst_new_file = os.path.join(dst, src_filename)
        if copy:
            shutil.copy2(src_old_file, dst_new_file)
        else:
            os.rename(src_old_file, dst_new_file)

def save_copy(src, dst):
    # type: (str, str) -> None
    logging.info('Saving copy from {} to {}'.format(src, dst))

    if os.path.isfile(src):
        shutil.copy2(src, dst) # Copy2 preserves metadata and accepts a dst directory
    elif os.path.isdir(src):
        if os.path.isdir(dst):
            move_files(src, dst)
        else:
            shutil.copytree(src, dst)
    else:
        raise FileExistsError("The source to copy does not exist")

def get_file_lines(file_path):
    # type: (str) -> list[str]
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def delete_files(path):
    # type: (str) -> None
    dirs = os.listdir(path)
    for file in dirs:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            logging.warning('Attempting to delete directory {}'.format(file_path))
            continue
        elif os.path.isfile(file_path):
            logging.info('Deleting file {}'.format(file_path))
            os.remove(file_path)

def rename_prefixes(dir, filename_with_desired_prefix, prefix_sep='.'):
    # type: (str, str, str) -> None
    desired_prefix = filename_with_desired_prefix.split(prefix_sep)[0]
    dir_filenames = os.listdir(dir)

    for filename in dir_filenames:
        old_prefix = filename.split(prefix_sep)[0]
        desired_new_suffix = filename[len(old_prefix):]
        desired_new_filename = desired_prefix + desired_new_suffix
        desired_new_filepath = os.path.join(dir, desired_new_filename)
        current_filepath = os.path.join(dir, filename)
        os.rename(current_filepath, desired_new_filepath)