'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from shutil                                               import copyfile
from package_parameters.parameters_visualization          import Parameters
import package_utils.fold_handler                         as fh
import os

# ----------------------------------------------------------------------------------------------------------------------FFF
def setParameters():

  p = Parameters(
                PDATA = '/home/laine/Desktop/CAROSEGDEEP_TEST/DATA_TEST/',                # PATH TO LOAD DATA
                PRES  = '/home/laine/Desktop/CAROSEGDEEP_TEST/DATABASE_VISUALIZATION',             # PATH TO SAVE DATABASE
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