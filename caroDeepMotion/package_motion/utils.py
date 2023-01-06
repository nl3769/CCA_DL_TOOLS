from icecream                               import ic

import numpy                                as np
import package_utils.signal_processing      as sp

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
def image_interpoland(I, interp_factor):
    """ Image interpolation x and z direction. """

    if len(I.shape) == 3:
        I = I.transpose(2, 0, 1)

    I = sp.image_interp_factor(I, interp_factor)

    if len(I.shape) == 3:
        I = I.transpose(2, 0, 1)

    return I

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
def scale_images(I1, I2, OF, roi_width, pixel_width, CF):

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
    OF_[..., 0] = image_interpoland(OF[..., 0], interp_factor)
    OF_[..., 1] = image_interpoland(OF[..., 1], interp_factor)
    OF_[..., 2] = image_interpoland(OF[..., 2], interp_factor)
    # ---  get the real pixel size after interpolation
    rCF, zcoef = compute_real_CF(Odim, Fdim, CF)
    # --- modify optical flow magnitude
    OF_[..., 0] *= Fdim[0] / Odim[0]
    OF_[..., 2] *= Fdim[1] / Odim[1]

    return I1, I2, OF_, rCF

# ----------------------------------------------------------------------------------------------------------------------
def get_patches(roi_width, roi_height, shift_x, shift_z, dim_img, substr, coordinates):

    # --- compute x-coordinates
    criterium = True
    pos_x = [0]
    while criterium:
        if pos_x[-1] + shift_x + roi_width < dim_img[1]:
            pos_x.append(pos_x[-1] + shift_x)
        else:
            pos_x.append(dim_img[1] - roi_width)
            criterium = False
    condition = True
    # --- compute z-coordinates
    criterium = True
    pos_z = [0]
    while criterium:
        if pos_z[-1] + shift_z + roi_height < dim_img[0] - 1:
            pos_z.append(pos_z[-1] + shift_z)
        else:
            pos_z.append(dim_img[0] - roi_height)
            criterium = False

    # --- get coordinates
    pos_x = np.asarray(pos_x)
    pos_z = np.asarray(pos_z)

    [X, Z] = np.meshgrid(pos_x, pos_z)
    id_patch = 1
    for id_x in range(X.shape[1]):
        for id_z in range(X.shape[0]):
            coordinates[substr + str(id_patch)] = (X[id_z][id_x], Z[id_z][id_x])
            id_patch += 1

    return coordinates

# ----------------------------------------------------------------------------------------------------------------------
def get_coordinates_full_img(shift_x, shift_z, roi_width, roi_height, dim_img):
    """ Compute patches position. """

    coordinates = {}
    substr = 'pos_'
    coordinates = get_patches(roi_width, roi_height, shift_z, shift_x, dim_img, substr, coordinates)

    return coordinates

# ----------------------------------------------------------------------------------------------------------------------
def patches_extraction(I1, I2, OF, coordinates, pixel_width, pixel_height, pairs_name):
    """ Get patch. """

    data = {
        'I1': [],
        'I2': [],
        'OF': [],
        'coord': [],
        'pname': []}

    for key in coordinates.keys():

        x, z = coordinates[key]
        if z+pixel_height-1 < I1.shape[0] and x+pixel_width-1 < I1.shape[1]:
            # --- extract on I1/I2
            data['I1'].append(I1[z:z+pixel_height, x:x+pixel_width])
            data['I2'].append(I2[z:z+pixel_height, x:x+pixel_width])
            data['coord'].append(coordinates[key])
            data['OF'].append(OF[z:z + pixel_height, x:x + pixel_width, :])
        else:
            ic('cannot process ' + key)

    return data