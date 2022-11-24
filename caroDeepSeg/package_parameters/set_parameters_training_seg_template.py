'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os

from shutil                                                     import copyfile
from package_parameters.parameters_training_seg                 import Parameters
import package_utils.fold_handler                               as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

  p = Parameters(
        PDATA='/home/laine/cluster/PROJECTS_IO/CARODEEPSEG/CUBS_DATABASE',                # PATH TO LOAD DATA
        PRES='/home/laine/cluster/PROJECTS_IO/CARODEEPSEG/',                      # PATH TO SAVE TRAINING RESULTS
        PSPLIT='/home/laine/cluster/PROJECTS_IO/CARODEEPSEG//SPLIT_PATIENT/fold_0',
        IN_VIVO =True,
        LEARNING_RATE=0.001,
        BATCH_SIZE=2,           # size of a batch
        NB_EPOCH=15,
        VALIDATION=True,
        DROPOUT=0.1,            # dropout during training
        WORKERS=4,
        USER='LAINE',
        EXPNAME='SEG_TRAINING_FOLD_0',
        DEVICE='cuda',          # cuda/cpu
        RESTORE_CHECKPOINT=True,
        ENTITY='nl37',
        # --- parameters for unet
        KERNEL_SIZE=(3, 3),
        PADDING=(1, 1),
        USE_BIAS=True,
        NGF=32,                 # number of input features of the Unet
        NB_LAYERS=4
  )

  pparam = os.path.join(p.PRES, p.EXPNAME, 'parameters')
  fh.create_dir(p.PRES)
  fh.create_dir(pparam)

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join(pparam, 'get_parameters_training.py'), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join(pparam, 'get_parameters_training.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p
