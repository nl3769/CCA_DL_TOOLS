import os
import argparse
import importlib

from package_GUI.mkClipArt          import mkClipArt
import matplotlib.pyplot            as plt
import pickle                       as pck
import package_utils.fold_handler   as pufh

# ----------------------------------------------------------------------------------------------------------------------
def load_info(path):
    """Takes path with .pkl extension and return data."""
    with open(path, 'rb') as f:
        loaded_dict = pck.load(f)

    return loaded_dict

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    p_img = os.path.join(p.PSAVE, 'images') # path to save the statistical image
    p_cf = os.path.join(p.PSAVE, 'CF') # path to save the calibration factor (pixel size) of the statistical image
    p_seg = os.path.join(p.PSAVE, 'SEG') # path the save the position of the interfaces

    pufh.create_dir(p_img)
    pufh.create_dir(p_cf)
    pufh.create_dir(p_seg)

    # --- make manual clip art
    IMC_density = load_info(p.PIMC)
    adventicia_density = load_info(p.PADVENTICIA)
    lumen_rayleigh = load_info(p.PLUMEN)
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
    """
    This function is used to create statistical images.
    """

    main()


