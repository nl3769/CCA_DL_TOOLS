from mat4py                 import loadmat
from PIL                    import Image
from PIL                    import ImageOps
import nibabel              as nib
import pickle               as pkl
import numpy                as np

# ----------------------------------------------------------------------------------------------------------------------
def load_mat(fname: str):
    """Load .mat file. """

    data = loadmat(fname)

    return data

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