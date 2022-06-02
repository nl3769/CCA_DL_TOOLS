'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_database import Parameters
import package_utils.fold_handler as fh
import os
from shutil import copyfile

# ****************************************************************
# *** HOWTO
# ****************************************************************

# 0) Do not modify this template file "setParameterstemplate.py"
# 1) Create a new copy of this file "setParametersTemplate.py" and rename it into "setParameters.py"
# 2) Indicate all the variables according to your local environment and experiment
# 3) Use your own "setParameters.py" file to run the code
# 4) Do not commit/push your own "setParameters.py" file to the collective repository, it is not relevant for other people
# 5) The untracked file "setParameters.py" is automatically copied to the tracked file "getParameters.py" for reproducibility
# ****************************************************************

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

  p = Parameters(
                PDATA='/home/laine/HDD/PROJECTS_IO/SIMULATION/CUBS',
                # PDATA='/run/media/laine/HDD/PROJECTS_IO/SIMULATION/CUBS',      # PATH TO LOAD DATA
                PRES='/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/TEST',             # PATH TO SAVE DATABASE
                ROI_WIDTH = 3e-3,                                                       # SIZE OF THE ROI WIDTH
                PIXEL_WIDTH = 128,                                                      # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW (IT IS EQUAL TO ROI_WIDTH)
                PIXEL_HEIGHT = 256,                                                     # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW
                SHIFT_X = 96,                                                           # X SHIFT TO GENERATE DATASET
                SHIFT_Z = 50                                                            # Z SHIFT TO GENERATE DATASET
  )

  fh.create_dir(p.PRES)

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(p.PRES, 'get_parameters_database.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join(p.PRES, 'get_parameters_database.py'), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join(p.PRES, 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p

# ----------------------------------------------------------------------------------------------------------------------