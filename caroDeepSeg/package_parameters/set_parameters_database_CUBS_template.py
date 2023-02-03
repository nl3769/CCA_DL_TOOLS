'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from shutil                                             import copyfile
from package_parameters.parameters_database             import Parameters
import os
import package_utils.fold_handler                       as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

    p = Parameters(
        PDATA           = '/home/laine/Documents/PROJECTS_IO/DATA/MEIBURGER/DATASET_for_CREATIS',              # PATH TO LOAD DATA
        PRES            = '/home/laine/Documents/PROJECTS_IO/CUBS_DATABASE_TEST_01',                                   # PATH TO SAVE DATABASE
        ROI_WIDTH       = 5e-3,                                                                 # SIZE OF THE ROI WIDTH
        PIXEL_WIDTH     = 256,                                                                  # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW (IT IS EQUAL TO ROI_WIDTH)
        PIXEL_HEIGHT    = 256,                                                                  # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW
        SHIFT_X         = 96,                                                                   # X SHIFT TO GENERATE DATASET
        SHIFT_Z         = 64                                                                    # Z SHIFT TO GENERATE DATASET
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