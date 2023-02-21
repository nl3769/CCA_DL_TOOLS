'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
from shutil                                         import copyfile
from package_parameters.parameters_get_distribution import Parameters
import package_utils.fold_handler                   as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

    p = Parameters(
        PIMAGES="/home/laine/Documents/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/images",
        PCF="/home/laine/Documents/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/CF",
        PLUMEN = "/home/laine/Documents/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/LUMEN_POSITION",
        PINTERFACES="/home/laine/Documents/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/SEG",
        PRES="/home/laine/Documents/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION",

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