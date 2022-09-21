from mat4py                 import loadmat

import imageio              as iio
import nibabel              as nib
import pickle               as pkl

# ----------------------------------------------------------------------------------------------------------------------
def load_mat(fname: str):
    """Load .mat file. """

    data = loadmat(fname)

    return data

# ----------------------------------------------------------------------------------------------------------------------
def load_image(fname: str):
    """Load image. """

    I = iio.imread(fname)

    return I

# ----------------------------------------------------------------------------------------------------------------------
def load_nii(fname: str):
    """Load .nii file. """

    data = nib.load(fname)

    return data

# ----------------------------------------------------------------------------------------------------------------------
def load_pickle(fname: str):
    """Load .pkl file. """

    with open(fname, 'rb') as f:
        data = pkl.load(f)

    return data

# ----------------------------------------------------------------------------------------------------------------------