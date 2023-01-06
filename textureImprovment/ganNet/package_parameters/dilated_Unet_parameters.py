import os

import torch.nn                         as nn

from package_parameters.inference_parameters      import Parameters
from shutil                             import copyfile
from package_utils.utils                import check_dir

def setParameters():

  p = Parameters(
    MODEL_NAME                 = 'dilatedUnet',
    PDATA                      = '/home/laine/Documents/PROJECTS_IO/DATA/GAN',
    DATABASE                   = {
        'training':   '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v02/training.txt',
        'validation': '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v02/validation.txt',
        'testing':    '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v02/testing.txt'
        },
    VALIDATION                 = True,
    LOSS                       = 'L2',
    lambda_GAN                 = 1/1000,
    lambda_pixel               = 1,
    LEARNING_RATE              = 0.0001,
    BATCH_SIZE                 = 2,
    NB_EPOCH                   = 500,
    IMAGE_NORMALIZATION        = (0, 255),
    KERNEL_SIZE                = (5, 5),
    PADDING                    = (2, 2),
    USE_BIAS                   = True,
    NGF                        = 64,
    NB_LAYERS                  = 3,
    IMG_SIZE                   = (256, 512),
    DROPOUT                    = 0,
    WORKERS                    = 4,
    EARLY_STOP                 = 100,
    OUTPUT_ACTIVATION          = None,
    PATH_RES                   = '/home/laine/Documents/PROJECTS_IO/GAN/dilatedUnet'
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
