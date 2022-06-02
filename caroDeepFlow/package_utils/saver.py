from PIL import Image
import nibabel as nib
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
def write_image(I, pres):
    """Write image in .png format. """

    im = Image.fromarray(I)
    im.save(pres, format="png")

# ----------------------------------------------------------------------------------------------------------------------
def write_nifty(I, pres):
    """Write image in .nii format. """

    img = nib.Nifti1Image(I, np.eye(4))
    nib.save(img, pres)

