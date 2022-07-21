import glob
import os
from scipy                  import interpolate
from mat4py                 import loadmat
import matplotlib.pyplot    as plt
import numpy                as np

def adapt_segmentation(dim_I, dim_org, LI, MA):

    dim_org = list(dim_org)
    grid_x_org = np.linspace(-dim_org[1]/2, dim_org[1]/2, dim_org[1])
    grid_x_I =  np.linspace(-dim_I[1]/2, dim_I[1]/2, dim_I[1])


    LI_new = {}
    MA_new = {}

    for patient in LI.keys():
        LI_ = LI[patient][1]
        LI_ = np.nan_to_num(LI_, nan=0)
        f = interpolate.Akima1DInterpolator(grid_x_org, LI_)
        LI_new[patient] = f(grid_x_I)

    for patient in MA.keys():
        MA_ = MA[patient][1]
        MA_ = np.nan_to_num(MA_, nan=0)
        f = interpolate.Akima1DInterpolator(grid_x_org, MA_)
        MA_new[patient] = f(grid_x_I)

    return LI_new, MA_new

# -----------------------------------------------------------------------------------------------------------------------
def add_annotation(LI, MA, seq):


    for id, key in enumerate(LI.keys()):

        for i in range(len(LI[key])):
            val = round(LI[key][i][0])
            if val > 0:
                seq[id][val, i] = 255

    for id, key in enumerate(MA.keys()):

        for i in range(len(MA[key])):
            val = round(MA[key][i][0])
            if val > 0:
                    seq[id][val, i] = 255

    return seq

# -----------------------------------------------------------------------------------------------------------------------
def compute_IMT(LI, MA, borders):


    IMT = np.zeros([int(borders['right_border'] - borders['left_border'] + 1), len(LI.keys())])

    for id, key in enumerate(LI.keys()):
        id_LI = np.where((LI[key][0] >= borders['left_border']) & (LI[key][0] <= borders['right_border']))
        id_MA = np.where((MA[key][0] >= borders['left_border']) & (MA[key][0] <= borders['right_border']))

        LI_ = LI[key][1][np.array(id_LI[0])]
        MA_ = MA[key][1][np.array(id_MA[0])]
        IMT[:, id] = MA_ - LI_

    return IMT

# -----------------------------------------------------------------------------------------------------------------------
def display_IMT(IMT):

     nb_img = IMT.shape[1]

     IMT_mean = np.zeros(nb_img)

     for id in range(nb_img):
         IMT_mean[id] = np.mean(IMT[:, id])

     plt.figure(1)
     plt.plot(IMT_mean)
     plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def get_seg_org(pdata, substr):

    # --- sort fname
    fnames = sorted(glob.glob(os.path.join(pdata, '*' + substr + '*')))
    LI = {}
    MA = {}

    for fname in fnames:

        LI[fname.split('/')[-1]] = load_interpolated_mat(fname, 'LI.mat')
        MA[fname.split('/')[-1]] = load_interpolated_mat(fname, 'MA.mat')

    return LI, MA

# ----------------------------------------------------------------------------------------------------------------------
def seg_interpolation(x, y):

    f = interpolate.Akima1DInterpolator(x, y)
    x_min = np.ceil(np.min(x))
    x_max = np.floor(np.max(x))

    x_new = np.linspace(x_min, x_max, int(x_max-x_min+1))
    y_new = f(x_new)

    # #######
    # plt.figure()
    # plt.plot(x_new, y_new)
    #
    # plt.figure()
    # plt.plot(x, y)
    # #######

    return x_new, y_new

# ----------------------------------------------------------------------------------------------------------------------
def load_interpolated_mat(pdata, fname):

    # ic(pdata)
    dir = os.path.join(pdata, 'phantom', fname)

    data = loadmat(dir)
    keys = list(data.keys())
    data = data[keys[0]]
    data = data['seg']

    x = np.linspace(0, len(data)-1, len(data))
    y = np.array(data)
    y = y - 1
    y [y<0] = 0
    # x, y = seg_interpolation(x, y)

    return [x, y]
