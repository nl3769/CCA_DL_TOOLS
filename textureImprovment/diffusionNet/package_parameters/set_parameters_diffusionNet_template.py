'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_diffusionNet         import Parameters
from shutil                                             import copyfile
import package_utils.fold_handler                       as fh
import os

def setParameters():

    p = Parameters(
        PDATA                       = "/home/laine/Documents/PROJECTS_IO/DATA/GAN",
        DATABASE                    = {
            'training'  : '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v01/training.txt',
            'validation': '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v01/validation.txt',
            'testing'   : '/home/laine/Documents/PROJECTS_IO/DATA/GAN/split_v01/testing.txt'
            },
        PRES                        = "/home/laine/Documents/PROJECTS_IO/DIFFUSION",
        VALIDATION                  = True,
        IMG_SIZE                    = (64, 64),
        IMAGE_NORMALIZATION         = (-1, 1),
        NB_LAYERS                   = 3,
        NGF                         = 32,
        KERNEL_SIZE                 = (3, 3),
        PADDING                     = (1, 1),
        USE_BIAS                    = True,
        INPUT_NC                    = 1,
        OUTPUT_NC                   = 1,
        TIME_EMB_DIM                = 32,
        TIME_STEP                   = 300,
        NB_EPOCH                    = 500,
        LEARNING_RATE               = 0.0001,
        BATCH_SIZE                  = 32,
        WORKERS                     = 0,
        EARLY_STOP                  = 50,
        BETA                        = [0.0001, 0.02]
    )

    pparam = os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)

    # --- Print all attributes in the console
    attrs = vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))

    # --- Return populated object from Parameters class
    return p

