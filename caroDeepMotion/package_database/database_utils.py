'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import math
import json
import matplotlib.pyplot                    as plt
import numpy                                as np
import package_utils.fold_handler           as fh
import package_utils.loader                 as ld
import package_utils.signal_processing      as sp
import package_utils.reader                 as rd
import package_utils.saver                  as ps
import pickle                               as pkl
import imageio                              as io

from icecream                               import ic

# ----------------------------------------------------------------------------------------------------------------------
def get_path_GIF(path_data, seq, id):
    """ Get path to create GIF. """
    path = {}

    path['I'] = []
    path['MA'] = []
    path['LI'] = []
    path['CF'] = []
    path['parameters'] = []
    for id_seq in seq:

        path['I'].append(path_data[id_seq]['path_image'])
        path['MA'].append(path_data[id_seq]['path_MA'])
        path['LI'].append(path_data[id_seq]['path_LI'])
        path['CF'].append(path_data[id_seq]['image_information'])
        path['parameters'].append(path_data[id_seq]['path_parameters'])

    return path

# ----------------------------------------------------------------------------------------------------------------------
def get_path(path_data, pairs, id):
    """ Get path. """
    path = {}

    path['CF'] = path_data[pairs[id][0]]['image_information']
    path['I1'] = path_data[pairs[id][0]]['path_image']
    path['I2'] = path_data[pairs[id][1]]['path_image']
    path['flow'] = path_data[pairs[id][1]]['path_flow']
    path['LI1'] = path_data[pairs[id][0]]['path_LI']
    path['LI2'] = path_data[pairs[id][1]]['path_LI']
    path['MA1'] = path_data[pairs[id][0]]['path_MA']
    path['MA2'] = path_data[pairs[id][1]]['path_MA']
    path['parameters'] = path_data[pairs[id][0]]['path_parameters']

    return path

# ----------------------------------------------------------------------------------------------------------------------
def get_seg(fname, key):
    """ Load the segmentation. """

    data, keys = ld.load_mat(fname)
    if key in keys:
        seg = np.squeeze(data[key]['seg'][0,0])
        seg[seg < 0] = 0
        seg = np.nan_to_num(seg, nan=0)
        seg = seg.squeeze()
    else:
        seg = None

    return seg

# ----------------------------------------------------------------------------------------------------------------------
def get_CF(fname):
    """ Load the pixel size of the image. """

    data, keys = ld.load_mat(fname)
    CF = data['image']['CF'][0, 0][0, 0]
    height = data['image']['height'][0, 0][0, 0]
    width = data['image']['width'][0, 0][0, 0]

    return CF, (height, width)

# ----------------------------------------------------------------------------------------------------------------------
def get_image(fname):
    """ Load image."""

    I = ld.load_image(fname)
    I = np.array(I)
    return I

# ----------------------------------------------------------------------------------------------------------------------
def get_flow(fname):
    """ Get optical flow ground truth. """

    data = ld.load_nii(fname)
    flow = data.get_fdata()
    flow = np.array(flow)
    return flow

# ----------------------------------------------------------------------------------------------------------------------
def get_interpolation_factor(oCF, roi_width, width_pixel):
    """ Compute interpolation factor to get the desired pixel size:
    CF_o: CF original
    roi_width: width in mm of the search region
    width_pixel: numbe rof pixels in the roi
    """

    dCF = roi_width / width_pixel
    factor = oCF / dCF

    return factor

# ----------------------------------------------------------------------------------------------------------------------
def load_data(path):
    """ Load data located in path directory. """

    zstart = get_param_value(path['parameters'], 'remove_top_region') * 2
    I1 = get_image(path['I1'])
    I2 = get_image(path['I2'])
    OF = get_flow(path['flow'])
    LI1 = get_seg(path['LI1'], 'LI_val') - zstart
    LI2 = get_seg(path['LI2'], 'LI_val') - zstart
    MA1 = get_seg(path['MA1'], 'MA_val') - zstart
    MA2 = get_seg(path['MA2'], 'MA_val') - zstart
    CF, seg_dim = get_CF(path['CF'])

    return I1, I2, OF, LI1, LI2, MA1, MA2, CF, seg_dim, zstart

# ----------------------------------------------------------------------------------------------------------------------
def load_cubs_data(paths):
    """ Load data located in path directory. """

    I = ld.load_image(paths['path_image'])
    LI = ld.load_mat(paths['path_LI'])
    MA = ld.load_mat(paths['path_MA'])

    LI = np.array(LI['seg']).squeeze()
    MA = np.array(MA['seg']).squeeze()

    with open(paths['image_information'], 'r') as f:
        CF = float(f.read()) * 1e-3

    return I, LI, MA, CF

# ----------------------------------------------------------------------------------------------------------------------
def load_prepared_data(paths):
    """ Load data located in path directory. """

    LI = ld.load_pickle(paths['path_LI'])
    MA = ld.load_pickle(paths['path_MA'])
    OF = ld.load_pickle(paths['path_field'])
    seq = ld.load_pickle(paths['path_image'])

    with open(paths['image_information'], 'r') as f:
        CF = float(f.read())

    return seq, OF, LI, MA, CF

# ----------------------------------------------------------------------------------------------------------------------
def load_data_GIF(path):
    """ Load data located in path directory. """

    I = []
    LI = []
    MA = []
    zstart = []
    CF = []
    seg_dim = []

    for id_path in path:
        if id_path == 'I':
            for key in path[id_path]:
                I.append(get_image(key))
        elif id_path == 'MA':
            for key in path[id_path]:
                MA.append(get_seg(key, 'MA_val'))
        elif id_path == 'LI':
            for key in path[id_path]:
                LI.append(get_seg(key, 'LI_val'))
        elif id_path == 'parameters':
            for key in path[id_path]:
                zstart.append(get_zstart(key))
        elif id_path == 'CF':
            for key in path[id_path]:
                CF_, seg_dim_ = get_CF(key)
                CF.append(CF_)
                seg_dim.append(seg_dim_)

    return I, LI, MA, CF, seg_dim, zstart

# ----------------------------------------------------------------------------------------------------------------------
def preprocessing(I1, I2, OF, LI1, LI2,  MA1, MA2, pairs, roi_width, pixel_width, CF, zstart):
    """ Preprocess data in order to extract the ROI corresponding. """

    Odim = I1.shape
    # --- check if dimension are consistent
    rd.check_image_dim(I1, I2, pairs)
    # --- adapt segmentation to image dimension
    LI1, MA1 = adapt_segmentation(LI1, MA1, Odim)
    LI2, MA2 = adapt_segmentation(LI2, MA2, Odim)
    # --- adapt optical flow dimension
    OF = adapt_optical_flow(OF, Odim, zstart, CF)
    # --- get interpolation factor and real pixel size of the interpolated image
    interp_factor = get_interpolation_factor(CF, roi_width, pixel_width)
    # --- interpolate image
    I1 = image_interpoland(I1, interp_factor)
    I2 = image_interpoland(I2, interp_factor)
    OF_ = np.zeros(I1.shape + (3,))
    OF_[..., 0] = image_interpoland(OF[...,0], interp_factor)
    OF_[..., 1] = image_interpoland(OF[..., 1], interp_factor)
    OF_[..., 2] = image_interpoland(OF[..., 2], interp_factor)
    # --- get size of the interpolated image
    Fdim = I1.shape
    # ---  get the real pixel size after interpolation
    rCF, zcoef = compute_real_CF(Odim, Fdim, CF)
    # --- adapt segmentation after interpolation
    LI1 *= zcoef
    MA1 *= zcoef
    LI2 *= zcoef
    MA2 *= zcoef
    LI1, MA1 = seg_interpoland(LI1, MA1, Fdim)
    LI2, MA2 = seg_interpoland(LI2, MA2, Fdim)
    # --- modify optical flow magnitude
    OF[..., 0] *= Odim[0]/Fdim[0]
    OF[..., 2] *= Odim[1] / Fdim[1]

    return I1, I2, OF_, LI1, LI2, MA1, MA2, rCF

# ----------------------------------------------------------------------------------------------------------------------
def preprocessing_prepared_data_cubs(I, LI, MA, roi_width, pixel_width, CF):
    """ Preprocess data in order to extract the ROI corresponding. """

    # --- get interpolation factor and real pixel size of the interpolated image
    interp_factor = get_interpolation_factor(CF, roi_width, pixel_width)

    # --- get size of original image
    Odim = I.shape

    # --- interpolate image
    I = image_interpoland(I, interp_factor)

    # --- get size of the interpolated image
    Fdim = I.shape

    # ---  get the real pixel size after interpolation
    rCF, zcoef = compute_real_CF(Odim, Fdim, CF)
    # --- adapt segmentation after interpolation
    LI *= zcoef
    MA *= zcoef
    LI, MA = seg_interpoland(LI, MA, Fdim)

    return I, LI, MA, rCF

# ----------------------------------------------------------------------------------------------------------------------
def preprocessing_prepared_data(I1, I2, OF, LI1, LI2,  MA1, MA2, roi_width, pixel_width, CF):
    """ Preprocess data in order to extract the ROI corresponding. """

    # --- get interpolation factor and real pixel size of the interpolated image
    interp_factor = get_interpolation_factor(CF, roi_width, pixel_width)
    # --- get size of original image
    Odim = I1.shape
    # --- interpolate image
    I1 = image_interpoland(I1, interp_factor)
    I2 = image_interpoland(I2, interp_factor)
    # --- get size of the interpolated image
    Fdim = I1.shape
    # --- interpolate flow
    OF_ = np.zeros(Fdim + (3,))
    OF_[..., 0] = image_interpoland(OF[...,0], interp_factor)
    OF_[..., 1] = image_interpoland(OF[..., 1], interp_factor)
    OF_[..., 2] = image_interpoland(OF[..., 2], interp_factor)
    # ---  get the real pixel size after interpolation
    rCF, zcoef = compute_real_CF(Odim, Fdim, CF)
    # --- adapt segmentation after interpolation
    LI1 *= zcoef
    MA1 *= zcoef
    LI2 *= zcoef
    MA2 *= zcoef
    # --- check if LI/MA exist else we do not interpolate the interfaces
    non_zeros = np.where(LI1 > 0)[0]
    if non_zeros.any():
        LI1, MA1 = seg_interpoland(LI1, MA1, Fdim)
        LI2, MA2 = seg_interpoland(LI2, MA2, Fdim)
    else:
        LI1, MA1 = None, None
        LI2, MA2 = None, None
    # --- modify optical flow magnitude
    OF_[..., 0] *= Fdim[0] / Odim[0]
    OF_[..., 2] *= Fdim[1] / Odim[1]

    return I1, I2, OF_, LI1, LI2, MA1, MA2, rCF

# ----------------------------------------------------------------------------------------------------------------------
def data_preparation_preprocessing(I1, I2, OF, LI1, LI2,  MA1, MA2, pairs, CF, zstart):
    """ Preprocess data in order to extract the ROI corresponding. """

    Odim = I1.shape
    # --- check if dimension are consistent
    rd.check_image_dim(I1, I2, pairs)
    # --- adapt segmentation to image dimension
    non_zeros = np.where(LI1 > 0)[0]
    if non_zeros.any():
        LI1, MA1 = adapt_segmentation(LI1, MA1, Odim)
        LI2, MA2 = adapt_segmentation(LI2, MA2, Odim)
    else:
        LI2, MA2 = None, None

    # --- adapt optical flow dimension
    OF = adapt_optical_flow(OF, Odim, zstart, CF)

    return I1, I2, OF, LI1, LI2, MA1, MA2

# ----------------------------------------------------------------------------------------------------------------------
def preprocessing_GIF(I, LI, MA):
    """ Preprocess data in order to extract the ROI corresponding. """

    Odim = I[0].shape
    LIo = []
    MAo = []

    # --- adapt segmentation to image dimension
    for i in range(len(I)):
        LIo_, MAo_ = adapt_segmentation(LI[i], MA[i], Odim)
        LIo.append(LIo_)
        MAo.append(MAo_)

    return LIo, MAo

# ----------------------------------------------------------------------------------------------------------------------
def compute_real_CF(Odim, Fdim, CF_init):
    """ Compute the real pixel size as we can't reach the desired value because of rounding. """

    zcoef = Fdim[0] / Odim[0]
    zCF = CF_init / zcoef

    xcoef = Fdim[1] / Odim[1]
    xCF = CF_init / xcoef

    CF = {"xCF": xCF,
          "zCF": zCF}

    return CF, zcoef

# ----------------------------------------------------------------------------------------------------------------------
def mean_pos(LI, MA):
    """ Compute avergae position between the LI and MA interface. """
    if LI is not None:
        mean = LI + MA
        mean = mean / 2
    else:
        mean = None

    return mean

# ----------------------------------------------------------------------------------------------------------------------
def adapt_segmentation(LI, MA, Odim):
    """ Adapt segmentation to the in-silico image (with adaptation). """

    LIdim = LI.shape
    MAdim = MA.shape

    grid_query = np.linspace(-Odim[1] / 2, Odim[1] / 2, Odim[1])
    grid_LI = np.linspace(-LIdim[0] / 2, LIdim[0] / 2, LIdim[0])
    grid_MA = np.linspace(-MAdim[0] / 2, MAdim[0] / 2, MAdim[0])

    LI = sp.signal_interpolation_1D(LI, grid_LI, grid_query)
    MA = sp.signal_interpolation_1D(MA, grid_MA, grid_query)

    LI = np.nan_to_num(LI, nan=0)
    MA = np.nan_to_num(MA, nan=0)

    return LI, MA

# ----------------------------------------------------------------------------------------------------------------------
def adapt_optical_flow(OF, Odim, zstart, CF):
    """ Fit optical flow to the in-silico image dimension. """

    # width, height, _ = OF.shape
    height, width, _ = OF.shape
    height_q, width_q = Odim

    x_q = np.linspace(- width_q/2 * CF, width_q/2 * CF, width_q)
    z_q = np.linspace(zstart, zstart + height_q * CF, height_q)
    # z_q = np.linspace(zstart, height * CF, height_q)

    x = np.linspace(- width / 2 * CF, width / 2 * CF, width)
    z = np.linspace(0, height * CF, height)

    X_q, Z_q = np.meshgrid(x_q, z_q)
    X, Z = np.meshgrid(x, z)

    OFo = np.zeros((height_q, width_q, 3))
    OFo[..., 0] = sp.grid_interpolation_2D(X_q, Z_q, OF[..., 0], X, Z)
    OFo[..., 1] = sp.grid_interpolation_2D(X_q, Z_q, OF[..., 1], X, Z)
    OFo[..., 2] = sp.grid_interpolation_2D(X_q, Z_q, OF[..., 2], X, Z)

    return OFo

# ----------------------------------------------------------------------------------------------------------------------
def seg_interpoland(LI, MA, Fdim):
    """ Adapt segmentation to the in-silico image (with adaptation). """

    LIdim = LI.shape
    MAdim = MA.shape


    grid_query = np.linspace(-LIdim[0] / 2, LIdim[0] / 2, Fdim[1])
    grid_LI = np.linspace(-LIdim[0] / 2, LIdim[0] / 2, LIdim[0])
    grid_MA = np.linspace(-MAdim[0] / 2, MAdim[0] / 2, MAdim[0])

    LI = sp.signal_interpolation_1D(LI, grid_LI, grid_query)
    MA = sp.signal_interpolation_1D(MA, grid_MA, grid_query)

    LI = np.nan_to_num(LI, nan=0)
    MA = np.nan_to_num(MA, nan=0)

    return LI, MA

# ----------------------------------------------------------------------------------------------------------------------
def get_roi_borders_cubs(LI, MA):
    """ Get roi borders. """

    roi_left, roi_right = [], []

    roi_left.append(LI.nonzero()[0][0])
    roi_left.append(MA.nonzero()[0][0])

    roi_right.append(LI.nonzero()[0][-1])
    roi_right.append(MA.nonzero()[0][-1])

    roi = {
        "left": max(roi_left) + 5,
        "right": min(roi_right) - 5}

    return roi

# ----------------------------------------------------------------------------------------------------------------------
def get_roi_borders(LI1, LI2, MA1, MA2):
    """ Get roi borders. """

    roi_left, roi_right = [], []
    if LI1 is not None:
        roi_left.append(LI1.nonzero()[0][0])
        roi_left.append(LI2.nonzero()[0][0])
        roi_left.append(MA1.nonzero()[0][0])
        roi_left.append(MA2.nonzero()[0][0])

        roi_right.append(LI1.nonzero()[0][-1])
        roi_right.append(LI2.nonzero()[0][-1])
        roi_right.append(MA1.nonzero()[0][-1])
        roi_right.append(MA2.nonzero()[0][-1])

        roi = {
            "left": max(roi_left) + 10,
            "right": min(roi_right) - 10}
    else:
        roi = {
            "left": None,
            "right": None}

    return roi

# ----------------------------------------------------------------------------------------------------------------------
def image_interpoland(I, interp_factor):
    """ Image interpolation x and z direction. """

    if len(I.shape) == 3:
        I = I.transpose(2,0,1)

    I = sp.image_interp_factor(I, interp_factor)

    if len(I.shape) == 3:
        I = I.transpose(2, 0, 1)

    return I

# ----------------------------------------------------------------------------------------------------------------------
def get_cropped_coordinates_cubs(roi_borders, pos, shift_x, shift_z, roi_width, roi_height):
    """ Compute position to get window. """

    nb_points = roi_borders['right'] - roi_borders['left']
    x = np.linspace(roi_borders['left'], roi_borders['right'], nb_points+1, dtype=int)

    coordinates = {}
    substr = 'pos_'
    criterium = True
    id = 0
    patch_id = 0
    while criterium:

        xc = x[id]
        zc0 = int(round(pos[xc] - roi_height / 2))
        zc1 = zc0 - shift_z
        zc2 = zc0 + shift_z

        if zc0 > 0:
            coordinates[substr + str(patch_id) + '_0'] = (xc, zc0)
        if zc1 > 0:
            coordinates[substr + str(patch_id) + '_1'] = (xc, zc1)
        if zc2 > 0:
            coordinates[substr + str(patch_id) + '_2'] = (xc, zc2)

        id += shift_x
        patch_id += 1

        # --- check criterium
        if x[id] + roi_width > roi_borders['right']:
            criterium = False


    return coordinates

# ----------------------------------------------------------------------------------------------------------------------
def get_rnd_patches(roi_width, roi_height, shift_x, shift_z, dim_img, substr, coordinates):

    pos_x = []
    condition = True
    incr = 0
    while condition:
        pos_x.append(incr*shift_x)
        if ((incr + 1) * shift_x + roi_width) > (dim_img[1] - roi_width):
            pos_x.append(dim_img[1] - roi_width - 1)
            condition = False
        incr+=1

    pos_z = []
    condition = True
    incr = 0
    while condition:
        pos_z.append(incr*shift_z)
        if ((incr + 1) * shift_z + roi_height) > (dim_img[0] - roi_height):
            pos_z.append(dim_img[0] - roi_height - 1)
            condition = False
        incr += 1

    pos_x = np.asarray(pos_x)
    pos_z = np.asarray(pos_z)

    [X, Z] = np.meshgrid(pos_x, pos_z)
    id_patch = 1
    for id_x in range(X.shape[1]):
        for id_z in range(X.shape[0]):
            coordinates[substr + str(id_patch)] = (X[id_z][id_x], Z[id_z][id_x])
            id_patch += 1

# ----------------------------------------------------------------------------------------------------------------------
def get_patch_carotid(pos, roi_height, coordinates, shift_z, x, shift_x, roi_width, roi_borders, dim_img):

    criterium = True
    patch_id = 0
    id = 0
    substr = 'pos_'
    default_pos = False

    while criterium:
        xc = x[id]
        zc0 = int(round(pos[xc] - roi_height / 2))
        zc1 = zc0 - shift_z
        zc2 = zc0 + shift_z

        if (zc0 + roi_height) < dim_img[0]:
            coordinates[substr + str(patch_id) + '_0'] = (xc, zc0)
        else:
            default_pos = True
        if (zc1 + roi_height) < dim_img[0]:
            coordinates[substr + str(patch_id) + '_1'] = (xc, zc1)
        else:
            default_pos = True
        if (zc2 + roi_height) < dim_img[0]:
            coordinates[substr + str(patch_id) + '_2'] = (xc, zc2)
        else:
            default_pos = True

        if default_pos:
            coordinates[substr + str(patch_id) + '_0'] = (xc, dim_img[0] - roi_height-1)
            default_pos = False

        id += shift_x
        patch_id += 1

        # --- check criterium
        if x[id] + roi_width >= roi_borders['right']:
            criterium = False

# ----------------------------------------------------------------------------------------------------------------------
def get_cropped_coordinates(roi_borders, pos1, pos2, shift_x, shift_z, roi_width, roi_height, dim_img):
    """ Compute position to get window. """

    if pos1 is not None:
        nb_points = roi_borders['right'] - roi_borders['left']
        x = np.linspace(roi_borders['left'], roi_borders['right'], nb_points+1, dtype=int)
        pos = (pos1 + pos2) / 2
    else:
        pos = None

    coordinates = {}
    substr = 'pos_'

    if pos is not None:
        get_patch_carotid(pos, roi_height, coordinates, shift_z, x, shift_x, roi_width, roi_borders, dim_img)
    else:
        get_rnd_patches(roi_width, roi_height, shift_x, shift_z, dim_img, substr, coordinates)

    return coordinates

# ----------------------------------------------------------------------------------------------------------------------
def get_param_value(fname, key):
    """ Get zstart. """

    # TODO: fname should not be a list
    extension = fname[0].split('.')[-1]
    if extension == "mat":
        data = ld.loadmat(fname)
        zstart = data['p'][key]

    elif extension == "json":
        with open(fname[0], 'r') as f:
            data = json.load(f)
        zstart = data[key]

    return zstart

# ----------------------------------------------------------------------------------------------------------------------
def data_extraction_cubs(LI, MA, I, coordinates, pixel_width, pixel_height):
    """ Get patch. """

    # --- get roi
    LIc = np.nonzero(LI)[0]
    MAc = np.nonzero(MA)[0]

    # --- mask creation
    mask = np.zeros(I.shape, dtype=np.uint8)
    for i in range(LIc.shape[0]):

        id = int(round(LIc[i]))
        LI_ = int(LI[id])
        MA_ = int(MA[id])

        mask[LI_:MA_, id] = 255

    data = {
        'I': [],
        'M': [],
        'pname': []}

    for id, key in enumerate(coordinates.keys()):

        x, z = coordinates[key]
        if z+pixel_height < I.shape[0] and x+pixel_width < I.shape[1]:
            # --- extract on I1/M1
            data['I'].append(I[z:z+pixel_height, x:x+pixel_width])
            data['M'].append(mask[z:z + pixel_height, x:x + pixel_width])

            data['pname'].append(key)

        else:
            ic('cannot process ' + key)

    return data

# ----------------------------------------------------------------------------------------------------------------------
def data_extraction(LI1, LI2, MA1, MA2, I1, I2, OF, coordinates, pixel_width, pixel_height, pairs_name):
    """ Get patch. """

    # --- get roi
    if LI1 is not None:
        LI1c = np.nonzero(LI1)[0]
        LI2c = np.nonzero(LI2)[0]
        MA1c = np.nonzero(MA1)[0]
        MA2c = np.nonzero(MA2)[0]

        rd.check_segmentation_dim(LI1c, LI2c, MA1c, MA2c, pairs_name)

        # --- mask creation
        mask1 = np.zeros(I1.shape, dtype=np.uint8)
        mask2 = np.zeros(I2.shape, dtype=np.uint8)
        for i in range(LI1c.shape[0]):

            id = int(round(LI1c[i]))
            LI1_ = int(LI1[id])
            MA1_ = int(MA1[id])
            LI2_ = int(LI2[id])
            MA2_ = int(MA2[id])

            mask1[LI1_:MA1_, id] = 255
            mask2[LI2_:MA2_, id] = 255

    data = {
        'I1': [],
        'M1': [],
        'I2': [],
        'M2': [],
        'OF': [],
        'pname': []}

    for id, key in enumerate(coordinates.keys()):

        x, z = coordinates[key]
        if z+pixel_height <= int(I1.shape[0]) and x+pixel_width < int(I1.shape[1]):
            # --- extract on I1/I2
            data['I1'].append(I1[z:z+pixel_height, x:x+pixel_width])
            data['I2'].append(I2[z:z + pixel_height, x:x + pixel_width])

            if LI1 is not None:
                # --- extract on M1/M2
                data['M1'].append(mask1[z:z + pixel_height, x:x + pixel_width])
                data['M2'].append(mask2[z:z + pixel_height, x:x + pixel_width])
            else:
                data['M1'].append(np.ones(data['I1'][0].shape))
                data['M2'].append(np.ones(data['I2'][0].shape))

            # --- extract on OF
            data['OF'].append(OF[z:z + pixel_height, x:x + pixel_width, :])

            data['pname'].append(key)

        else:
            ic('cannot process ' + key)

    return data

# ----------------------------------------------------------------------------------------------------------------------
def adapt_seg_borders_cubs(LI, MA, roi_borders):
    """ Adapt segmentation: fit with 0 values strictly lower than the then border and strictly higher than the right border. """

    width = LI.shape[0]

    for i in range(0, roi_borders['left']):
        LI[i] = 0
        MA[i] = 0

    for i in range(roi_borders['right'], width):
        LI[i] = 0
        MA[i] = 0

    return LI, MA

# ----------------------------------------------------------------------------------------------------------------------
def adapt_seg_borders(LI1, LI2, MA1, MA2, roi_borders, width):
    """ Adapt segmentation -> fit with 0 values strictly lower than the then border and strictly higher than the right border. """

    if LI1 is not None:
        set_zero_left = np.arange(0, roi_borders['left'])
        set_zero_right = np.arange(roi_borders['right'], LI1.shape[0])

        LI1[set_zero_left] = 0
        LI2[set_zero_left] = 0
        MA1[set_zero_left] = 0
        MA2[set_zero_left] = 0

        LI1[set_zero_right] = 0
        LI2[set_zero_right] = 0
        MA1[set_zero_right] = 0
        MA2[set_zero_right] = 0

    return LI1, LI2, MA1, MA2

# ----------------------------------------------------------------------------------------------------------------------
def save_data_cubs(data, CF, pres, pname):
    """ Save data
    -> I1, I2: patch than contains the intima-media complex (.png format)
    -> M1, M2: masks of the IMC for segmentation task (.png format)
    -> OF: displacment field between I1 and I2 (.nii format)
    """

    psave = os.path.join(pres, pname)
    fh.create_dir(psave)

    keys = list(data.keys())

    for key in keys:
        if 'pname' not in key:
            fh.create_dir(os.path.join(psave, key))

    # --- get number of patches
    nb_patches = len(data[keys[0]])
    debug = False

    for id in range(nb_patches):
        for key in keys:
            if key != 'pname':
                psave_ = os.path.join(psave, key, data['pname'][id] + ".pkl")
                ps.write_pickle(data[key][id], psave_)

                if debug == True:
                    if 'M' in key:
                        M = data[key][id]
                    if 'I' in key:
                        I = data[key][id]

        if debug == True:
            pres = os.path.join('/home/laine/Desktop/patch_res/', pname.split('/')[:1][0])
            fh.create_dir(pres)
            debug_plot_patch(M, I, os.path.join(pres, data['pname'][id] + '_pos1.png'))

        with open(os.path.join(psave, "CF.txt"), "w") as f:
            for key in CF.keys():
                f.write(key + " " + str(CF[key]) + '\n')

# ----------------------------------------------------------------------------------------------------------------------
def save_data(data, CF, pres, pname, dim):
    """ Save data
    -> I1, I2: patch than contains the intima-media complex (.png format)
    -> M1, M2: masks of the IMC for segmentation task (.png format)
    -> OF: displacment field between I1 and I2 (.pkl format)
    """

    psave = os.path.join(pres, pname)
    fh.create_dir(psave)

    keys = list(data.keys())

    for key in keys:
        if 'pname' not in key:
            fh.create_dir(os.path.join(psave, key))

    # --- get number of patches
    nb_patches = len(data[keys[0]])
    debug = False

    for id in range(nb_patches):
        bak_pres = []
        condition = True
        for key in keys:
            if key != 'pname':

                psave_ = os.path.join(psave, key, data['pname'][id] + ".pkl")
                bak_pres.append(psave_)
                ps.write_pickle(data[key][id], psave_)
                dim_ = data[key][id].shape
                if len(dim_) == 3:
                    dim_ = dim_[:-1]

                if dim_ != dim:
                    condition = False

        if condition == False:
            for path in bak_pres:
                print('remove: ', path)
                os.remove(path)

                if debug == True:
                    if 'M1' in key:
                        M1 = data[key][id]
                    if 'M2' in key:
                        M2 = data[key][id]
                    if 'I1' in key:
                        I1 = data[key][id]
                    if 'I2' in key:
                        I2 = data[key][id]
                    if 'OF' in key:
                        motion = data[key][id]

        if debug == True:
            pres = os.path.join('/home/laine/Desktop/patch_res/', pname.split('/')[:1][0])
            fh.create_dir(pres)
            debug_plot_patch_v2(M1, M2, I1, I2, np.linalg.norm(motion[..., ::2], axis=2), os.path.join(pres, data['pname'][id] + '.png'))
            # debug_plot_patch(M2, I2, os.path.join(pres, data['pname'][id] + '_pos2.png'))

        with open(os.path.join(psave, "CF.txt"), "w") as f:
            for key in CF.keys():
                f.write(key + " " + str(CF[key]) + '\n')

# ----------------------------------------------------------------------------------------------------------------------
def debug_plot_patch(M, I, pname):

    plt.figure()
    plt.subplot2grid((1, 3), (0, 0), colspan=1)
    plt.imshow(I, cmap='gray')
    plt.axis('off')
    plt.subplot2grid((1, 3), (0, 1), colspan=1)
    plt.imshow(M, cmap='gray')
    plt.axis('off')
    plt.subplot2grid((1, 3), (0, 2), colspan=1)
    superimpose = np.ones(I.shape + (3,))
    superimpose[...,0], superimpose[..., 1], superimpose[..., 2] = I, I, I
    mulmtx = (1 - M)
    superimpose[..., 0] = np.multiply(superimpose[..., 0], mulmtx)
    superimpose[..., 2] = np.multiply(superimpose[..., 2], mulmtx)
    superimpose[..., 1] = superimpose[..., 1]
    superimpose = superimpose.astype(np.int)
    plt.imshow(superimpose)
    plt.axis('off')
    plt.savefig(pname)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def debug_plot_patch_v2(M1, M2, I1, I2, motion, pname):

    plt.figure()
    plt.subplot2grid((3, 2), (0, 0), colspan=1)
    plt.imshow(I1, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.subplot2grid((3, 2), (0, 1), colspan=1)
    plt.imshow(M1, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    plt.subplot2grid((3, 2), (1, 0), colspan=1)
    plt.imshow(I2, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.subplot2grid((3, 2), (1, 1), colspan=1)
    plt.imshow(M2, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    plt.subplot2grid((3, 2), (2, 0), colspan=1)
    plt.imshow(I2 - I1, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    plt.subplot2grid((3, 2), (2, 1), colspan=1)
    plt.imshow(motion, cmap='hot')
    plt.colorbar()
    plt.axis('off')

    plt.savefig(pname)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def save_data_preparation(I_seq, OF_seq, LI_seq, MA_seq, CF, pres, pname):
    """ Save data in pickle format. It saves images, boundaries (LI and MA), calibration factor (CF: pixel size), and the displacement field. """

    fh.create_dir(os.path.join(pres, pname))
    # --- save images
    with open(os.path.join(pres, pname, "images-" + pname + ".pkl"), 'wb') as f:
        pkl.dump(I_seq, f)
    # --- save displacement field
    with open(os.path.join(pres, pname, "displacement_field-" + pname + ".pkl"), 'wb') as f:
        pkl.dump(OF_seq, f)
    # --- save LI boundary
    with open(os.path.join(pres, pname, "LI-" + pname + ".pkl"), 'wb') as f:
        pkl.dump(LI_seq, f)
    # --- save MA boundary
    with open(os.path.join(pres, pname, "MA-" + pname + ".pkl"), 'wb') as f:
        pkl.dump(MA_seq, f)
    # --- save calibration factor
    with open(os.path.join(pres, pname, "CF-" + pname + ".txt"), 'w') as f:
        f.write(str(CF))

    # # --- we can save a mat version to be used in matlab -> Set saveMat to True if you need.
    saveMat = False
    if saveMat:
        import scipy.io
        fh.create_dir(os.path.join(pres, pname, 'mat_files'))
        scipy.io.savemat(os.path.join(pres, pname, 'mat_files', "images-" + pname + ".mat"), {'data': I_seq})
        scipy.io.savemat(os.path.join(pres, pname, 'mat_files', "displacement_field-" + pname + ".mat"), {'data': OF_seq})
        scipy.io.savemat(os.path.join(pres, pname, 'mat_files', "LI-" + pname + ".mat"), {'data': LI_seq})
        scipy.io.savemat(os.path.join(pres, pname, 'mat_files', "MA-" + pname + ".mat"), {'data': MA_seq})

# ----------------------------------------------------------------------------------------------------------------------
def add_annotation(I, LI, MA):
    """ For debugging purposes only. This function adds boundaries to the images. """
    nbi = len(I)
    I_a = []

    for i in range(nbi):
        I_ = I[i]
        LI_ = LI[i]
        MA_ = MA[i]
        I_ = np.repeat(I_[:, :, np.newaxis], 3, axis=2)

        id_LI = np.nonzero(LI_)[0]
        id_MA = np.nonzero(MA_)[0]

        for i in range(id_LI.shape[0]):
            I_[round(LI_[id_LI[i]]), id_LI[i], 0] = 0
            I_[round(LI_[id_LI[i]]), id_LI[i], 1] = 100
            I_[round(LI_[id_LI[i]]), id_LI[i], 2] = 0

        for i in range(id_MA.shape[0]):
            I_[round(MA_[id_MA[i]]), id_MA[i], 0] = 0
            I_[round(MA_[id_MA[i]]), id_MA[i], 1] = 0
            I_[round(MA_[id_MA[i]]), id_MA[i], 2] = 100

        I_a.append(I_)

    return I_a

# ----------------------------------------------------------------------------------------------------------------------
def mk_animation(pres, pname, CF):

    with open(os.path.join(pres, pname, "images-" + pname + ".pkl"), 'rb') as f:
        I_seq = pkl.load(f)

    with open(os.path.join(pres, pname, "LI-" + pname + ".pkl"), 'rb') as f:
        LI_seq = pkl.load(f)

    with open(os.path.join(pres, pname, "MA-" + pname + ".pkl"), 'rb') as f:
        MA_seq = pkl.load(f)

    with open(os.path.join(pres, pname, "displacement_field-" + pname + ".pkl"), 'rb') as f:
        OF = pkl.load(f)


    if not math.isnan(LI_seq[0, 1]) and not math.isnan(MA_seq[0, 1]):
        for id_frame in range(I_seq.shape[-1]):
            for i in range(I_seq.shape[1]):
                if int(LI_seq[i, id_frame]) > 0:
                    I_seq[int(LI_seq[i, id_frame]), i, id_frame] = 255
                if int(MA_seq[i, id_frame]) > 0:
                    I_seq[int(MA_seq[i, id_frame]), i, id_frame] = 255

    images = []
    for id_seq in range(I_seq.shape[-1]):
        images.append(I_seq[..., id_seq].astype(np.uint8))

    io.mimsave(os.path.join(pres, pname, "sequence.gif"), images, fps=10)

    flow = []
    for id_seq in range(OF.shape[-1]):
        motion = OF[..., id_seq].copy()
        motion *= 255
        flow.append(motion.astype(np.uint8))

    io.mimsave(os.path.join(pres, pname, "OF.gif"), flow, fps=10)

# ----------------------------------------------------------------------------------------------------------------------