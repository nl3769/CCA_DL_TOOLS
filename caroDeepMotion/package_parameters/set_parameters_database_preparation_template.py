'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os

from shutil                                                 import copyfile
from package_parameters.parameters_database_preparation     import Parameters

import package_utils.fold_handler                           as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

    p = Parameters(
        PDATA           = '/run/media/laine/DISK/PROJECTS_IO/SIMULATION/IMAGENET',                                      # PATH TO LOAD DATA
        PRES            = '/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/prepared_data_IMAGENET',        # PATH TO SAVE DATABASE
        )

    pparam = os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)
    # --- Print all attributes in the console
    attrs = vars(p)
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_database.py'))

    # --- Return populated object from Parameters class
    return p

# ----------------------------------------------------------------------------------------------------------------------
