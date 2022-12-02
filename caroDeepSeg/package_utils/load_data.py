"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import scipy.io
import cv2
import os
import matplotlib.pyplot    as plt
import numpy                as np
import package_utils.loader as pul

# ----------------------------------------------------------------------------------------------------------------------
def load_data(path: str, param):
    """ Load any type of date (mat, dicom, tiff) and returns the interpolated image and the scale coefficient. """

    extension = path.split('.')
    LOI = False

    noneDicom = ["mat", "tiff", "png", "nii"]
    if extension[-1] not in noneDicom:
        extension = 'dicom'
    else:
        extension = extension[-1]

    if path.split('/')[-1].split('_')[0] == 'LOI':
        LOI = True
    if extension == 'tiff' or extension == 'png':
        return load_TIFF_image(path=path, param=param)

# ----------------------------------------------------------------------------------------------------------------------
def load_TIFF_image(path: str, param):
    """ Load tiff image and returns the interpolated image and the scale coefficient. """

    seq = pul.load_image(path)
    if np.max(seq) == 1:
        seq = seq * 255

    path_spacing = os.path.join(param.PCF, path.split('/')[-1].split('.')[0] + "_CF.txt")
    CF_org = read_CF_directory(path_spacing)

    I_in = seq.copy()

    I_out, scale, CF = sequence_preprocessing(
        seq=np.expand_dims(seq, axis=0),
        width_roi_meter = param.ROI_WIDTH,
        width_roi_pixel = param.PIXEL_WIDTH,
        height_roi_pixel = param.PIXEL_HEIGHT,
        CF = CF_org
    )

    return I_out, I_in, scale, CF, CF_org

# ----------------------------------------------------------------------------------------------------------------------

def read_CF_directory(path: str):
    """ Read calibration factor. """

    f = open(path, "r")
    val = f.readline().split(' \n')
    
    # --- convert in meter
    return float(val[0]) * 1e-3

# ----------------------------------------------------------------------------------------------------------------------
def sequence_preprocessing(seq: str, width_roi_meter: float, width_roi_pixel: int, height_roi_pixel, CF: float):
    """ Sequence preprocessing -> vertical interpolation to reach a vertical pixel size of spatial_resolution. """

    [nb_frame, height, width] = seq.shape
    CF_out = width_roi_meter/width_roi_pixel

    scale_x = (width_roi_pixel * CF) / width_roi_meter
    scale_y = (height_roi_pixel * CF) / (height_roi_pixel * CF_out)

    height_out = round(height * scale_y)
    width_out = round(width * scale_x)
    out = np.zeros((nb_frame, height_out, width_out))

    for i in range(nb_frame):
        out[i, :, :] = cv2.resize(seq[i, :, :].astype(np.float32), (width_out, height_out), interpolation=cv2.INTER_LINEAR)

    return out, {'scale_x': scale_x, 'scale_y': scale_y}, {'CF_x': (CF * width)/width_out, 'CF_y': (CF * height)/height_out}

# ----------------------------------------------------------------------------------------------------------------------
def load_borders(path: str):
    """ Loads borders in .mat file and return the left and the right borders. """

    mat_b = scipy.io.loadmat(path)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0]
    left_b = mat_b['border_left']
    left_b = left_b[0, 0]

    return left_b - 1, right_b - 1

# ----------------------------------------------------------------------------------------------------------------------
def load_tiff(sequence: str, PATH_TO_CF: str):
    """Loads a tiff image. For the CUBS database, each .tiff has an external .txt file that contains the pixel size (CF).
    Original image and the pixel size are returned."""

    seq = plt.imread(sequence)
    if len(seq.shape) == 3:
        seq = seq[:, :, 0]
    path_spacing = os.path.join(PATH_TO_CF, sequence.split('/')[-1].split('.')[0] + "_CF.txt")
    spatial_res_y = read_CF_file(path_spacing)

    return spatial_res_y, seq

# ----------------------------------------------------------------------------------------------------------------------
def read_CF_file(path: str):
    """ Loads pixel size in .txt file for CUBS database. """

    f = open(path, "r")
    val = f.readline().split(' \n')
    return float(val[0]) / 1000

# ----------------------------------------------------------------------------------------------------------------------
def load_annotation(path: str, patient: str, nameExpert: str):
    """ Loads annotation stored in .mat file. The function returns two vectors corresponding to LI and MA interfaces."""

    path_IFC3 = path + '/' + patient.split('.')[0] + "_IFC3_" + nameExpert + ".mat"
    path_IFC4 = path + '/' + patient.split('.')[0] + "_IFC4_" + nameExpert + ".mat"

    IFC3 = scipy.io.loadmat(path_IFC3)
    IFC3 = IFC3['seg']
    IFC4 = scipy.io.loadmat(path_IFC4)
    IFC4 = IFC4['seg']

    return IFC3, IFC4

# ----------------------------------------------------------------------------------------------------------------------
def get_files(path: str):
    """ Returns a list containing the name of all the files. """

    file = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                file.append(entry.name)
    return file

# ----------------------------------------------------------------------------------------------------------------------
