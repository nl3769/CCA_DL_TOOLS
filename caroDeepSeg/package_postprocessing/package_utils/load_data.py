import os

import imageio          as iio
import numpy            as np
import pickle           as pkl
from mat4py             import loadmat

# ----------------------------------------------------------------------------------------------------------------------
def load_mat(fname: str):
    """Load .mat file. """

    data = loadmat(fname)

    return data

# ----------------------------------------------------------------------------------------------------------------------
def load_image(fname: str):
    """Load image. """

    I = iio.imread(fname)
    I = np.array(I)

    return I

# ----------------------------------------------------------------------------------------------------------------------
def load_seg(fname, key):
    """ Load the segmentation. """

    data = loadmat(fname)
    seg = data[key]['seg']
    seg = np.array(seg)
    seg[seg < 0] = 0
    seg = np.nan_to_num(seg, nan=0)
    seg = seg.squeeze()

    return seg

# ----------------------------------------------------------------------------------------------------------------------
def load_CF(path):
    I = load_mat(path)
    CF = I['image']['CF']
    
    return CF

# ----------------------------------------------------------------------------------------------------------------------
def get_seg(pseg, LI):
    
    seg_dic = {}

    for key in LI:
        idx = []
        pos = []
        path = os.path.join(pseg, key)
        with open(path, 'r') as f:
            seg = f.readlines()
        
        for val in seg:
            idx.append(int(val.split(' ')[0]))
            pos.append(float(val.split(' ')[-1].split('\n')[0]))

        seg_dic[key] = [idx, pos]
    
    return seg_dic
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def load_pickle(path):
    
    with open(path, 'rb') as f:
        x = pkl.load(f)

    return x
