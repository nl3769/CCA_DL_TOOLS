import imageio  as iio
import numpy    as np
from mat4py import loadmat

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
