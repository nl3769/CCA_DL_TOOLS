import glob
import os
from package_utils.data_handler     import load_interpolated_mat
import scipy.io                     as sio
import numpy                        as np

# ----------------------------------------------------------------------------------------------------------------------
def get_seg(pdata, substr):

    # --- sort fname
    fnames = sorted(glob.glob(os.path.join(pdata, '*_id_*')))
    LI = {}
    MA = {}

    for fname in fnames:

        LI[fname.split('/')[-1]] = load_interpolated_mat(fname, 'LI.mat')
        MA[fname.split('/')[-1]] = load_interpolated_mat(fname, 'MA.mat')

    return LI, MA

# -----------------------------------------------------------------------------------------------------------------------
def get_borders(LI, MA):

    x_min = []
    x_max = []

    for key in LI.keys():

        x_min.append(np.nonzero(LI[key][1])[0][0])
        x_min.append(np.nonzero(MA[key][1])[0][0])
        x_max.append(np.nonzero(LI[key][1])[0][-1])
        x_max.append(np.nonzero(MA[key][1])[0][-1])

    x_min = np.array(x_min)
    x_max = np.array(x_max)

    x_min = np.max(x_min)
    x_max = np.min(x_max)

    return {'left_border': x_min, 'right_border':x_max}

# ----------------------------------------------------------------------------------------------------------------------
def load_carolab_res(pres, roi_left, roi_right):

    data = sio.loadmat(pres)
    seg = data['data'][0, 0]['seg']
    right_border = data['data'][0, 0]['right_border']-1
    left_border = data['data'][0, 0]['left_border']-1

    left_diff  = int(roi_left - left_border)
    right_diff = int(right_border - roi_right)

    test = seg[:, left_diff:right_diff , :] - 1

    carolab = {}

    for i in range(test.shape[-1]):
        name = 'seq_' + str(i)
        carolab[name] = {}
        carolab[name]['LI'] = test[0, :, i]
        carolab[name]['MA'] = test[1, :, i]
        carolab[name]['IMT'] = carolab[name]['MA'] - carolab[name]['LI']

    return carolab