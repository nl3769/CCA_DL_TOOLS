"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import glob
import sys
from icecream import ic

# ----------------------------------------------------------------------------------------------------------------------
def check_segmentation_dim(LI1, LI2, MA1, MA2, iname):
    """ Check if dimension are consistent. """

    flag = False
    if LI1[0] == LI2[0] and LI1[0] == MA1[0] and LI1[0] == MA2[0] and flag == False:
        # --- nothing to do
        flag = False
    else:
        flag = True
    if LI1[-1] == LI2[-1] and LI1[-1] == MA1[-1] and LI1[-1] == MA2[-1] and flag == False:
        # --- nothing to do
        flag = False
    else:
        flag = True
    if flag:
        sys.exit(f'Error in check check_segmentation_dim | pair 0 {iname[0]} | pair 1 {iname[1]}')

# ----------------------------------------------------------------------------------------------------------------------
def check_image_dim(I0, I1, pname):
    """ Check if dimension are consistent. """

    I0height, I0width = I0.shape
    I1height, I1width = I1.shape
    flag = False
    if (I0height != I1height):
        flag = True
    if (I0width != I1width):
        flag = True
    if flag:
        ic(pname)
        sys.exit(f'Error in check_image_dim {pname}')

# ----------------------------------------------------------------------------------------------------------------------
def get_fname(dir: str, sub_str: str, fold = None) -> str:
    """ Get file name according to substring. """

    folder = '' if fold is None else fold
    folder = os.path.join(dir, folder)
    if type(sub_str) == list:
        files = []
        for fname in sub_str:
            file = glob.glob(os.path.join(folder, '*' + fname + '*'))
            if len(file)==1:
                files.append(file)
    else:
        files = glob.glob(os.path.join(folder, '*' + sub_str + '*'))
    if len(files) != 1:
        ic(folder)
        ic(sub_str)
        sys.exit('Error in get_fname')

    return files[0]

# ----------------------------------------------------------------------------------------------------------------------
def read_txt(path):
    """ Read .txt file. """

    with open(path, 'r') as f:
        data = f.readlines()

    return data

# ----------------------------------------------------------------------------------------------------------------------