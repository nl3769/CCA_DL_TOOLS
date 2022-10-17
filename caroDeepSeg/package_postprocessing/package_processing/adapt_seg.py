import numpy                as np
from scipy                  import interpolate

# ----------------------------------------------------------------------------------------------------------------------
def adapt_segmentation(LI, MA, Odim):
    """ Adapt segmentation to the in-silico image (with adaptation). """

    LIdim = LI.shape
    MAdim = MA.shape

    grid_query  = np.linspace(-Odim[1] / 2, Odim[1] / 2, Odim[1])
    grid_LI     = np.linspace(-LIdim[0] / 2, LIdim[0] / 2, LIdim[0])
    grid_MA     = np.linspace(-MAdim[0] / 2, MAdim[0] / 2, MAdim[0])

    LI = signal_interpolation_1D(LI, grid_LI, grid_query) - 1
    MA = signal_interpolation_1D(MA, grid_MA, grid_query) - 1
    LI[LI == -1] = 0
    MA[MA == -1] = 0
    LI = np.nan_to_num(LI, nan=0)
    MA = np.nan_to_num(MA, nan=0)

    return LI, MA

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
