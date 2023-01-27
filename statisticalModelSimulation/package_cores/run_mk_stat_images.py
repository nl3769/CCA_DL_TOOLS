import os

from package_GUI.mkClipArt          import mkClipArt
from PIL import Image
import numpy                        as np
import matplotlib.pyplot            as plt
import pickle                       as pck
import package_utils.fold_handler   as pufh

# ----------------------------------------------------------------------------------------------------------------------
def load_info(path):
    '''Takes path with .pkl extension and return data.'''
    with open(path, 'rb') as f:
        loaded_dict = pck.load(f)

    return loaded_dict

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- set parameters and create directories
    p_IMC = "/home/laine/Documents/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/stat_model/IMC.pkl"
    p_adventicia = "/home/laine/Documents/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/stat_model/IMC.pkl"
    p_lumen = "/home/laine/Documents/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/stat_model/lumen.pkl"
    psave = "/home/laine/Desktop/statistical_images"

    p_img = os.path.join(psave, 'images') # path to save the statistical image
    p_cf = os.path.join(psave, 'CF') # path to save the calibration factor (pixel size) of the statistical image
    p_seg = os.path.join(psave, 'SEG') # path the save the position of the interfaces

    pufh.create_dir(p_img)
    pufh.create_dir(p_cf)
    pufh.create_dir(p_seg)

    # --- make manual clip art
    IMC_density = load_info(p_IMC)
    adventicia_density = load_info(p_adventicia)
    lumen_rayleigh = load_info(p_lumen)
    scale_rayleigh = lumen_rayleigh['full']['scale_parameters']

    nb_images = 10 # number of images to process
    for i in range(nb_images):
        img = mkClipArt(window_name='test', clip_size=(450, 620, 3))
        I, cf, interfaces = img(IMC_density, adventicia_density, scale_rayleigh)

        # --- save results
        plt.imsave(os.path.join(p_img, 'image_' + str(i) + '.tiff'), I, cmap='gray')
        with open(os.path.join(p_cf, 'image_' + str(i) + '_CF.txt'), 'w') as f:
            f.write(str(cf * 1e3))
        with open(os.path.join(p_seg, 'image_' + str(i) + '_IFC3.txt'), 'w') as f:
            for id, pos in enumerate(interfaces["LI_bottom"]):
                f.write(str(id) + " " + str(pos) + "\n")
        with open(os.path.join(p_seg, 'image_' + str(i) + '_IFC4.txt'), 'w') as f:
            for id, pos in enumerate(interfaces["MA_bottom"]):
                f.write(str(id) + " " + str(pos) + "\n")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()


