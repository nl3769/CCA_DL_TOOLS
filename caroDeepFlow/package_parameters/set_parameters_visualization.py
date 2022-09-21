'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from shutil                                               import copyfile
from package_parameters.parameters_visualization          import Parameters
import package_utils.fold_handler                         as fh
import os

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

def setParameters():

  p = Parameters(
                PDATA='/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/DATABASE',                # PATH TO LOAD DATA
                PRES='/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/DATASET/VISUALIZATION_TEST',             # PATH TO SAVE DATABASE
  )

  pparam = os.path.join(p.PRES, 'parameters')
  fh.create_dir(p.PRES)
  fh.create_dir(pparam)

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_visualization.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join(pparam, 'get_parameters_visualization.py'), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join(pparam, 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p