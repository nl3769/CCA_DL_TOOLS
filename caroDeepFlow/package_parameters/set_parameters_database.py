'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from shutil                                             import copyfile
from package_parameters.parameters_database             import Parameters
import os
import package_utils.fold_handler                       as fh
# ****************************************************************
# *** HOWTO
# ****************************************************************

# 0) Do not modify this template file "setParameterstemplate.py"
# 1) Create a new copy of this file "setParametersTemplate.py" and rename it into "setParameters.py"
# 2) Indicate all the variables according to your local environment and experiment
# 3) Use your own "setParameters.py" file to package_core the code
# 4) Do not commit/push your own "setParameters.py" file to the collective repository, it is not relevant for other people
# 5) The untracked file "setParameters.py" is automatically copied to the tracked file "getParameters.py" for reproducibility
# ****************************************************************

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

    p = Parameters(
        PDATA           ='/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/PREPARED_DATABASE',             # PATH TO LOAD DATA
        PRES            ='/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/DATABASE',    # PATH TO SAVE DATABASE
        ROI_WIDTH       = 6e-3,                                                                 # SIZE OF THE ROI WIDTH
        PIXEL_WIDTH     = 256,                                                                  # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW (IT IS EQUAL TO ROI_WIDTH)
        PIXEL_HEIGHT    = 256,                                                                  # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW
        SHIFT_X         = 32,                                                                   # X SHIFT TO GENERATE DATASET
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