'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import cv2
import numpy as np
from scipy import interpolate

# ----------------------------------------------------------------------------------------------------------------------
def image_interp_x(I, factor):
    """ Interpolate image in x-direction. """

    if len(I.shape) == 3:
        _, height, width = I.shape
    else:
        height, width = I.shape
    width_interp = int(np.round(width * factor))
    dsize = (width_interp, height)
    I = cv2.resize(I, dsize, interpolation=cv2.INTER_LINEAR)

    return I

# ----------------------------------------------------------------------------------------------------------------------
def image_interp_y(I, factor):
    """ Interpolate image in y-direction. """

    if len(I.shape) == 3:
        _, height, width = I.shape
    else:
        height, width = I.shape
    height_interp = int(np.round(height * factor))
    dsize = (width, height_interp)
    I = cv2.resize(I, dsize, interpolation=cv2.INTER_LINEAR)

    return I

# ----------------------------------------------------------------------------------------------------------------------
def signal_interpolation_1D(s, x_org, x_query, mode = 'linear'):
    """ 1D signal interpolation. """


    if mode == "linear":
        F = interpolate.interp1d(x_org, s)
        out = F(x_query)
        nonzero = out.nonzero()
        out[nonzero[0][0]] = 0
        out[nonzero[0][-1]] = 0

    elif mode == "akima":
        F = interpolate.Akima1DInterpolator(x_org, s)


    return out

# ----------------------------------------------------------------------------------------------------------------------
def grid_interpolation_2D(Xq, Zq, V, X, Z):
    """ Interpolate 2D grid data. """

    dimq = Xq.shape
    pointq = (Xq.flatten(), Zq.flatten())
    pointo = (X.flatten(), Z.flatten())
    val = V.flatten()
    out = interpolate.griddata(pointo, val, pointq, method='linear')
    out = out.reshape(dimq)

    return out

# ----------------------------------------------------------------------------------------------------------------------