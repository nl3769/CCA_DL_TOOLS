'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_postProcessInSilico  import Parameters
from shutil                                             import copyfile
import package_utils.fold_handler                       as fh
import os

def setParameters():

    p = Parameters(
        PDATA           = "/home/laine/Documents/PROJECTS_IO/SIMULATION/cluster_res/tech_002",
        PRES            = "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING",
        PMODEL          = "/home/laine/tux/JEAN_ZAY/RESULTS/GAN/GAN/GAN:.1_pxl:1_lss:L2_ker_7_7-hist:0-20/model/model_validation.pth",
        DIM_IMG_GAN     = (256, 256),
        INTERVAL        = (0, 20),
        NB_LAYERS       = 5,
        NGF             = 32,
        KERNEL_SIZE     = (7, 7),
        PADDING         = (3, 3),
        USE_BIAS        = True
        )

    pparam = os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)

    # --- Print all attributes in the console
    attrs = vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))

    # --- Return populated object from Parameters class
    return p

