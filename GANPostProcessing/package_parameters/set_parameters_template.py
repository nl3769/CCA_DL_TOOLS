from package_parameters.parameters import Parameters
import os
from shutil import copyfile

def setParameters():

  p = Parameters( PATH_KITTY = '/home/laine/Documents/PROJECTS_IO/DATA/OPTICAL_FLOW/KITTY',
                  MODEL_NAME = 'RAFT')

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('parameters', os.path.basename(__file__)), os.path.join('parameters', 'cp_' + os.path.basename(__file__)))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join('parameters', 'get_parameters.py'), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join('parameters', 'get_parameters.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p