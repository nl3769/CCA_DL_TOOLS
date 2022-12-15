import numpy as np
from scipy import interpolate

# ----------------------------------------------------------------------------------------------------------------------
def warpper(OF, I1):

    xof = OF[..., 0]
    zof = OF[..., 2]

    [height, width] = I1.shape

    x_org = np.linspace(-(width-1)/2, (width-1)/2, width)
    z_org = np.linspace(-(height-1)/2, (height-1)/2, height)

    [X_org, Z_org] = np.meshgrid(x_org, z_org)

    X_warpped = X_org + xof
    Z_warpped = Z_org + zof

    X_warpped = X_warpped.flatten()
    Z_warpped = Z_warpped.flatten()

    I1_warpped = interpolate.griddata((Z_warpped, X_warpped), I1.flatten(), (Z_org, X_org), method='cubic', fill_value=0)

    return I1_warpped

# ----------------------------------------------------------------------------------------------------------------------
def motion_norm(OF):
    """ Compute L2 norm. """

    OFx, OFy, OFz = OF[..., 0], OF[..., 1], OF[..., 2]


# ----------------------------------------------------------------------------------------------------------------------
def motion_args(OF):
    """ Compute argument in degree. """

    OFx, OFy, OFz = OF[..., 0], OF[..., 1], OF[..., 2]