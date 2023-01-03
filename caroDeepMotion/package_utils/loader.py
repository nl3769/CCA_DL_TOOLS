from PIL                    import Image
from PIL                    import ImageOps
import nibabel              as nib
import pickle               as pkl
import numpy                as np
import scipy.io             as io

# ----------------------------------------------------------------------------------------------------------------------
def load_mat(fname: str):
    """Load .mat file. """
    data = io.loadmat(fname)
    # data = loadmat(fname)
    keys = list(data.keys())
    keys = [key for key in keys if '__' not in key]

    return data, keys

# ----------------------------------------------------------------------------------------------------------------------
def load_image(fname: str):
    """Load image. """

    I = Image.open(fname)
    if I.mode == 'RGB':
        I = ImageOps.grayscale(I)
    I = np.array(I)

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
