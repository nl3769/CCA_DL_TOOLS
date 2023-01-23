import numpy                as np
from PIL                    import Image, ImageOps

# ----------------------------------------------------------------------------------------------------------------------
def load_image(fname: str):
    """Load image. """

    I = Image.open(fname)
    if I.mode == 'RGB':
        I = ImageOps.grayscale(I)
    I = np.array(I)

    return I

# ----------------------------------------------------------------------------------------------------------------------
def load_pos(path):

    with open(path, 'r') as f:
        data = f.readlines()

    data = [key.replace('\n', '') for key in data]
    x = [int(key.split(' ')[0]) for key in data]
    z = [float(key.split(' ')[1]) for key in data]

    return [x, z]