'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_mk_stat_images import Parameters
import os
from shutil import copyfile
import package_utils.fold_handler as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

    p = Parameters(
        PIMC="/run/media/laine/DISK/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/MK_MODEL/stat_model/IMC.pkl",
        PADVENTICIA="/run/media/laine/DISK/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/MK_MODEL/stat_model/IMC.pkl",
        PLUMEN="/run/media/laine/DISK/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/MK_MODEL/stat_model/lumen.pkl",
        PSAVE="/run/media/laine/DISK/PROJECTS_IO/STATISTICAL_MODEL_SIMULATION/STATISTICAL_IMAGES")

    pparam = os.path.join(p.PSAVE, 'backup_parameters')
    fh.create_dir(pparam)
    # --- Print all attributes in the console
    attrs = vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')
    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))
    # --- Return populated object from Parameters class
    return p