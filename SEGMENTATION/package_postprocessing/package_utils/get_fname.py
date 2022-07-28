import os
import glob
import sys

#  ----------------------------------------------------------------------------------------------------------------------
def get_fname(dir: str, sub_str: str, fold = None) -> str:
    """ Get file name according to substring. """

    folder = '' if fold is None else fold
    folder = os.path.join(dir, folder)
    files = glob.glob(os.path.join(folder, '*' + sub_str + '*'))

    if len(files) != 1:
        ic(folder)
        ic(sub_str)
        sys.exit('Error in get_fname')

    return files[0]
