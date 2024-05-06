import functools
import glob
import os
import shutil
import numpy as np


def remkdir(dirname: str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def _lstrip_filepath(lstrip_str: str, filename: str) -> str:
    strip_len = len(lstrip_str)
    return filename[strip_len:]


def find_by_format(folder_path: str, pathname_pattern: str) -> tuple[np.ndarray, np.ndarray]:
    name_files = np.array(glob.glob(folder_path + pathname_pattern, recursive=True))
    name_files_trimmed = np.array(list(map(functools.partial(_lstrip_filepath, folder_path), name_files)))

    return name_files, name_files_trimmed
