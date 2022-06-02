import numpy as np
from scipy import interpolate

# ----------------------------------------------------------------------------------------------------------------------
def warpper(OF, I1):

    xof = OF[..., 0]
    zof = OF[..., 2]

    [height, width] = I1.shape

    x_org = np.linspace(0, width-1, width)
    z_org = np.linspace(0, height-1, height)

    [X_org, Z_org] = np.meshgrid(x_org, z_org)

    X_warpped = X_org.flatten() + xof.flatten()
    Z_warpped = Z_org.flatten() + zof.flatten()

    I1_warpped = interpolate.griddata((Z_warpped, X_warpped), I1.flatten(), (Z_org, X_org), method='linear', fill_value=0)

    return I1_warpped

# ----------------------------------------------------------------------------------------------------------------------
def flow_norm(OF):
    """ Compute L2 norm. """

    OFx, OFy, OFz = OF[...,0], OF[...,1], OF[...,2]


# ----------------------------------------------------------------------------------------------------------------------
def flow_args(OF):
    """ Compute argument in degree. """

    OFx, OFy, OFz = OF[..., 0], OF[..., 1], OF[..., 2]