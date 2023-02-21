import argparse
import importlib
import os
import numpy                        as np
import package_utils.fold_handler   as fh

from package_GUI.LumenDetection     import getLumen
from PIL                        import Image

# ----------------------------------------------------------------------------------------------------------------------
def save_seg(seg, path):

    x = seg[0]
    z = seg[1]
    with open(path, 'w') as f:
        for id in range(seg[0].shape[0]):
            val = str(int(x[id])) + " " + str(z[id]) + "\n"
            f.write(val)

# ----------------------------------------------------------------------------------------------------------------------
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# ----------------------------------------------------------------------------------------------------------------------
def data_loader(fname):

    # --- get extension
    ext = fname.split('.')[-1]

    if 'tiff' in ext:
        I = load_image(fname)
    else:
        raise Exception("No implementation for this type of data.")

    return I

# ----------------------------------------------------------------------------------------------------------------------
def load_image(path):

    I = Image.open(path)
    I = np.array(I)
    if I.shape[-1] == 3:
        I = I[..., 0]

    return I

# ----------------------------------------------------------------------------------------------------------------------
def load_seg(path):

    with open(path, 'r') as f:
        data = f.readlines()
    data = [key.replace('\n', '') for key in data]
    id = [int(key.split(' ')[0]) for key in data]
    pos = [float(key.split(' ')[1]) for key in data]

    return id, pos

# ----------------------------------------------------------------------------------------------------------------------
def main():
    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()
    # --- create pres
    fh.create_dir(p.PRES)
    # --- list files
    patients = os.listdir(p.PIMAGES)
    patients.sort()
    seg_files = os.listdir(p.PDATASEG)
    # --- loop over images
    for patient in patients:
        wname = patient.split('.')[0]
        seg_name = []
        for key in seg_files:
            if wname in key:
                seg_name.append(key)

        I = data_loader(os.path.join(p.PIMAGES, patient))
        id_LI, pos_LI = load_seg(os.path.join(p.PDATASEG, seg_name[0]))
        id_MA, pos_MA = load_seg(os.path.join(p.PDATASEG, seg_name[1]))
        I = np.repeat(I[:, :, np.newaxis], 3, axis=2)
        handler = getLumen(wname, I, id_LI, pos_LI, id_MA, pos_MA)
        top_val, bottom_val = handler()

        if len(top_val) > 0:
            patient_name = patient.split('.')[0]
            save_seg(top_val, os.path.join(p.PES, patient_name + '_lumen_top.txt'))
            save_seg(bottom_val, os.path.join(p.PES, patient_name + '_lumen_bottom.txt'))

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This function calls a home made GUI for lumen detection. It has to be run before run_get_distribution.py
    """

    main()

# ----------------------------------------------------------------------------------------------------------------------