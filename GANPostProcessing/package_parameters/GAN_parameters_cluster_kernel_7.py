from parameters.parameters import Parameters
import os
from shutil import copyfile
from utils.utils import check_dir
16
# ****************************************************************
# *** HOWTO
# ****************************************************************
# 0) Do not modify this template file "setParameterstemplate.py"
# 1) Create a new copy of this file "setParametersTemplate.py" and rename it into "setParameters.py"
# 2) Indicate all the variables according to your local environment and experiment
# 3) Use your own "setParameters.py" file to run the code
# 4) Do not commit/push your own "setParameters.py" file to the collective repository, it is not relevant for other people
# 5) The untracked file "setParameters.py" is automatically copied to the tracked file "getParameters.py" for reproductibility
# ****************************************************************

def setParameters():

  p = Parameters(MODEL_NAME = 'GAN_UPSAMPLE',
                 PDATA  = '/home/laine/PROJECTS_IO/DATA/GAN',
                 DATASET = {'training': '/home/laine/PROJECTS_IO/DATA/GAN/folds/training.txt',
                            'validation': '/home/laine/PROJECTS_IO/DATA/GAN/folds/validation.txt',
                            'testing': '/home/laine/PROJECTS_IO/DATA/GAN/folds/testing.txt'},
                 VALIDATION=True,
                 LEARNING_RATE = 0.001,
                 BATCH_SIZE = 16,
                 NB_EPOCH = 5,
                 NORMALIZATION=True,
                 KERNEL_SIZE=5,
                 IMG_SIZE = (256, 512),
                 DROPOUT = 0.1,
                 WORKERS = 4,
                 EARLY_STOP=60,
                 PATH_RES='/home/laine/PROJECTS_IO/POSTPROCESSING/GAN/KERNEL_7')

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  check_dir(p.PATH_RES)
  copyfile(os.path.join('parameters', os.path.basename(__file__)), os.path.join(p.PATH_RES, 'cp_' + os.path.basename(__file__)))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join(p.PATH_RES, 'cp_' + os.path.basename(__file__)), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join(p.PATH_RES, 'cp_' + os.path.basename(__file__)), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p
