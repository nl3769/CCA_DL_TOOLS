from PIL import Image

import nibabel                              as nib
import numpy                                as np
import pickle                               as pkl

# ----------------------------------------------------------------------------------------------------------------------
def write_image(I, pres):
    ''' Write image in .png format. '''

    im = Image.fromarray(I)
    im.save(pres, format="png")

# ----------------------------------------------------------------------------------------------------------------------
def write_nifty(I, pres):
    ''' Write image in .nii format. '''

    img = nib.Nifti1Image(I, np.eye(4))
    nib.save(img, pres)

# ----------------------------------------------------------------------------------------------------------------------
def write_pickle(data, pres):
    ''' Write image in .pkl format. '''

    with open(pres, 'wb') as f:
        pkl.dump(data, f)

