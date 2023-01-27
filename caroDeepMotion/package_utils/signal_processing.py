'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from PIL import Image
import numpy as np
from scipy import interpolate

# ----------------------------------------------------------------------------------------------------------------------
def image_interp_factor(I, factor):
    ''' Interpolate image in x-direction. '''

    if len(I.shape) == 3:
        _, height, width = I.shape
    else:
        height, width = I.shape

    height_q = round(height * factor)
    width_q = round(width * factor)

    I = Image.fromarray(I)
    im1 = I.resize((width_q, height_q))
    im1 = np.array(im1)

    return im1

# ----------------------------------------------------------------------------------------------------------------------
def signal_interpolation_1D(s, x_org, x_query, mode = 'linear'):
    ''' 1D signal interpolation. '''

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
    ''' Interpolate 2D grid data. '''

    dimsq = Xq.shape
    pointsq = (Xq.flatten(), Zq.flatten())
    pointso = (X.flatten(), Z.flatten())
    val = V.flatten()
    out = interpolate.griddata(pointso, val, pointsq, method='linear')
    out = out.reshape(dimsq)

    return out

# ----------------------------------------------------------------------------------------------------------------------