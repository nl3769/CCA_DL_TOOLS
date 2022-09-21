import os

import torch.nn                         as nn

from package_parameters.parameters_inference      import Parameters
from shutil                                       import copyfile
from package_utils.utils                          import check_dir

def setParameters():

  p = Parameters(
    MODEL_NAME                 = 'dilatedUnet',
    PDATA                      = '/home/laine/Documents/PROJECTS_IO/DATA/GAN',
    PMODEL                     = '/home/laine/tux/JEAN_ZAY/RESULTS/GAN/GAN/dilatedUnet_L2_255_upconv_00/model',
    DATABASE                   = {
        'training':   '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_00/training.txt',
        'validation': '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_00/validation.txt',
        'testing':    '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_00/testing.txt'
        },
    NB_SAVE                    = 30,
    IMAGE_NORMALIZATION        = (0, 255),
    IMG_SIZE                   = (256, 512),
    PATH_RES                   = '/home/laine/Desktop/GAN_inference',
    KERNEL_SIZE                = (5, 5),
    PADDING                    = (2, 2),
    USE_BIAS                   = True,
    UPCONV                     = True,
    NGF                        = 64,
    NB_LAYERS                  = 4,
    OUTPUT_ACTIVATION          = None
    )

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  path_param = os.path.join(p.PATH_RES, 'parameters_backup')
  check_dir(path_param)
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(path_param, 'backup_' + os.path.basename(__file__)))

  # --- Return populated object from Parameters class
  return p
