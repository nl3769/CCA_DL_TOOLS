import os

from package_parameters.parameters      import Parameters
from shutil                             import copyfile
from package_utils.utils                import check_dir

def setParameters():

  p = Parameters(
      MODEL_NAME                 = 'UnetGZ',
      PDATA                      = '/home/laine/PROJECTS_IO/DATA/GAN',
      DATABASE                   = {
          'training'  : '/home/laine/PROJECTS_IO/DATA/GAN/split_v01/training.txt',
          'validation': '/home/laine/PROJECTS_IO/DATA/GAN/split_v01/validation.txt',
           'testing'   : '/home/laine/PROJECTS_IO/DATA/GAN/split_v01/testing.txt'},
      VALIDATION                 = True,
      LOSS                       = 'L2',
      lambda_GAN                 = 1/30,
      lambda_pixel               = 1,
      LEARNING_RATE              = 0.0001,
      BATCH_SIZE                 = 4,
      NB_EPOCH                   = 300,
      IMAGE_NORMALIZATION        = (0, 10),
      KERNEL_SIZE                = (3, 3),
      USE_BIAS                   = True,
      NGF                        = 32,               # number of input features of the Unet 
      NB_LAYERS                  = 5,
      IMG_SIZE                   = (256, 256),
      DROPOUT                    = 0,
      WORKERS                    = 0,
      EARLY_STOP                 = 50,
      PATH_RES                   = '/home/laine/PROJECTS_IO/GAN/lbd-GAN:1-over-30_lbd-pxl:1_kernel:3-3_loss:L2')

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
