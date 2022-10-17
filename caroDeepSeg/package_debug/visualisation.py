import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
import numpy as np


def display_image(I, id_fig = None, title = None):

    id_fig = 0 is id_fig is None
    title = "" is title is None

    plt.figure(id_fig)
    plt.imshow(I, cmap="gray")
    plt.title(title)

# ----------------------------------------------------------------------------------------------------------------------
def add_annotation(I, LI, MA, roi=None):

    ROI = {"left": 0, "right": I.shape[1]} if roi is None else roi
    I = np.repeat(I[..., np.newaxis], 3, axis=2)

    for id in range(ROI['left'], ROI['right']):

        zli = int(round(LI[id]))
        zma = int(round(MA[id]))

        I[zli, id, 0] = 255
        I[zli, id, 1] = 0
        I[zli, id, 2] = 0

        I[zma, id, 0] = 255
        I[zma, id, 1] = 0
        I[zma, id, 2] = 0

    plt.figure()
    plt.imshow(I)
    plt.title("add annotation")