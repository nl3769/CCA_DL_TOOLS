import os
import torch.nn                         as nn
from package_parameters.parameters      import Parameters
from shutil                             import copyfile
from package_utils.utils                import check_dir

def setParameters():

  p = Parameters(
    MODEL_NAME                 = 'SRGan',                                                                               # name of the model (unet/SGan/dilatedUnet)
    PDATA                      = '/home/laine/Documents/PROJECTS_IO/DATA/GAN',                                          # path to load data
    DATABASE                   = {                                                                                      # path to .txt containing patients' names for training/validation/test
        'training': '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v01/training.txt',
        'validation': '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v01/validation.txt',
        'testing': '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v01/testing.txt'
        },
    VALIDATION = True,                                                                                                  # use validation subset
    RESTORE_CHECKPOINT = False,                                                                                         # restore checkpoint (true/false)
    LOSS = 'L2',                                                                                                        # loss (L1/L2/L1L2/histo_loss)
    lambda_GAN = 1/1000,                                                                                                # weight for loss's discriminator
    lambda_pixel = 1,                                                                                                   # weight for loss's generator
    LEARNING_RATE = 0.0001,                                                                                             # learning rate value
    BATCH_SIZE = 2,                                                                                                     # batch size
    NB_EPOCH = 500,                                                                                                     # number of epoch for training
    IMAGE_NORMALIZATION = (0, 1),                                                                                       # interval for image normalization
    DATA_AUG = False,                                                                                                   # data augmentation (true/false)
    KERNEL_SIZE = (5, 5),                                                                                               # size of the kernel
    PADDING = (2, 2),                                                                                                   # padding size
    USE_BIAS = True,                                                                                                    # use bias (true/false)
    UPCONV = True,                                                                                                      # upconvolution (if false then upsampling is applied)
    NGF = 32,                                                                                                           # number of input filter of the network (Unet)
    NB_LAYERS = 5,                                                                                                      # depth of Unet
    IMG_SIZE = (256, 256),                                                                                              # size of the input images
    DROPOUT = 0,                                                                                                        # dropout value
    WORKERS = 4,                                                                                                        # number of worker
    EARLY_STOP = 100,                                                                                                   # early stop value - stop the training if overfitting
    OUTPUT_ACTIVATION = nn.ReLU(),                                                                                      # output activation layer
    PATH_RES = '/home/laine/Desktop/GAN/SRGan',                                                                         # path to save training result
    USE_WB = False                                                                                                      # use w&b (true/false)
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