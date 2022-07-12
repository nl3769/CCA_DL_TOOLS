import os

from package_parameters.parameters          import Parameters
from shutil                                 import copyfile
from package_utils.utils                    import check_dir

def setParameters():

  p = Parameters(MODEL_NAME         = 'AI4HEALTH',
                 PDATA              = '/gpfswork/rech/obh/uby91ul/PROJECTS_IO/DATA/GAN',
                 DATASET            = {'training'  : '/gpfswork/rech/obh/uby91ul/PROJECTS_IO/DATA/GAN/split_v01/training.txt',
                                       'validation': '/gpfswork/rech/obh/uby91ul/PROJECTS_IO/DATA/GAN/split_v01/validation.txt',
                                       'testing'   : '/gpfswork/rech/obh/uby91ul/PROJECTS_IO/DATA/GAN/split_v01/testing.txt'},
                 VALIDATION         = True,
                 LOSS               = 'L2',
                 LOSS_BALANCE       = {'lambda_GAN': 1, 'lambda_pixel': 10},
                 CASCADE_FILTERS    = 2,
                 LEARNING_RATE      = 0.0001,
                 BATCH_SIZE         = 8,
                 NB_EPOCH           = 500,
                 NORMALIZATION      = True,
                 KERNEL_SIZE        = 5,
                 BILINEAR           = False,
                 IMG_SIZE           = (256, 512),
                 DROPOUT            = 0,
                 WORKERS            = 4,
                 EARLY_STOP         = 150,
                 PATH_RES           = '/gpfswork/rech/obh/uby91ul/PROJECTS_IO/POSTPROCESSING/GAN/GAN_UP_CONV_TEST')

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  check_dir(p.PATH_RES)
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(p.PATH_RES, 'cp_' + os.path.basename(__file__)))

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

