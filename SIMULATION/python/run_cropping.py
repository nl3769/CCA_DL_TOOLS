import os
import imageHandler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mat4py import loadmat

def save_seg(path, seg):

    with open(path, 'w') as f:
        for id in range(seg.shape[0]):

            z = seg[id]
            if z < 0:
                z = 0
            x = id

            val = str(x) + " " + str(z) + "\n"
            f.write(val)

# ----------------------------------------------------------------------------------------------------------------------
def load_seg(path):

    seg = loadmat(path)
    seg = seg['seg']
    seg = np.array(seg)
    seg = seg.squeeze()

    return seg

# ----------------------------------------------------------------------------------------------------------------------
def save_image(path, I):

    rawtiff = Image.fromarray(I)
    rawtiff.save(path)

# ----------------------------------------------------------------------------------------------------------------------
def load_image(path):

    I = Image.open(path)
    I = np.array(I)

    if I.shape[-1] == 3:
        I = I[..., 0]

    return I

# ----------------------------------------------------------------------------------------------------------------------
def list_files(path):
    files = os.listdir(path)

    return files

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pdataI = '/home/laine/Documents/PROJECTS_IO/SIMULATION/PREPROCESSING/DATA_IMAGE'
    pdataSeg = '/home/laine/Documents/PROJECTS_IO/SIMULATION/PREPROCESSING/DATA_SEG'
    presI = '/home/laine/Documents/PROJECTS_IO/SIMULATION/PREPROCESSING/RESULTS_IMAGE'
    presSeg = '/home/laine/Documents/PROJECTS_IO/SIMULATION/PREPROCESSING/RESULTS_SEG'
    patients = list_files(pdataI)
    seg_files = list_files(pdataSeg)

    for patient in patients:

        wname = patient.split('.')[0]
        seg_name = []

        for key in seg_files:
            if wname in key:
                seg_name.append(key)

        I = load_image(os.path.join(pdataI, patient))
        seg1 = load_seg(os.path.join(pdataSeg, seg_name[0]))
        seg2 = load_seg(os.path.join(pdataSeg, seg_name[1]))
        I = np.repeat(I[:, :, np.newaxis], 3, axis=2)
        handler = imageHandler.getBorders(wname, I, seg1, seg2)
        I_cropped, seg1, seg2 = handler()

        if I_cropped is not None:
            os.remove(os.path.join(pdataI, patient))
            save_seg(os.path.join(presSeg, seg_name[0].replace('mat', 'txt')), seg1)
            save_seg(os.path.join(presSeg, seg_name[1].replace('mat', 'txt')), seg2)
            save_image(os.path.join(presI, patient), I_cropped)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
