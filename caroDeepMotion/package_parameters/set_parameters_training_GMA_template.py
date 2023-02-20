'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_training_GMA import Parameters
import package_utils.fold_handler as fh
import os
from shutil import copyfile

def setParameters():

    p = Parameters(
        MODEL_NAME                  = 'gma',                                                                            # chose gma or raft
        PDATA = '/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/database_training_IMAGENET',                # PATH TO LOAD DATA
        PRES = '/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/NETWORK_TRAINING',                                                                           # PATH TO SAVE TRAINING RESULTS
        PSPLIT = '/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/SPLIDATA',
        LEARNING_RATE=0.0001,
        BATCH_SIZE                  = 1,                                                                        # size of a batch
        NB_EPOCH                    = 500,
        VALIDATION                  = True,
        DROPOUT                     = 0.0,                                                                      # dropout during training
        GAMMA                       = 0.8,                                                                      # see later what it is
        ADD_NOISE                   = False,                                                                    # see later what it is
        CORRELATION_LEVEL           = 4,                                                                        # see later what it is
        CORRELATION_RADIUS          = 4,                                                                        # see later what it is
        NB_ITERATION                = 12,
        ALTERNATE_COORDINATE        = False,                                                                    # see later what it is
        WORKERS                     = 0,
        POSITION_ONLY               = False,
        POSITION_AND_CONTENT        = False,
        NUM_HEAD                    = 8,
        CONTEXT_DIM                 = 128,
        HIDDEN_DIM                  = 128,
        ADVENTICIA_DIM              = 1,                                                                        # part of adventitia in mm
        USER                        = 'LAINE',
        EXPNAME                     = 'GMA_TRAINING_TEST',
        DEVICE                      = 'cuda',                                                                   # cuda/cpu
        RESTORE_CHECKPOINT          = False,                                                      # shared or split
        SYNTHETIC_DATASET           = True,
        KERNEL_SIZE                 = (3, 3),
        PADDING                     = (1, 1),
        USE_BIAS                    = True,
        USE_WANDB=False,  # use W&B
        NGF                         = 32,  # number of input features of the Unet
        NB_LAYERS                   = 4)

    pparam = os.path.join(p.PRES, p.EXPNAME, 'backup_parameters')
    fh.create_dir(pparam)

    # --- Print all attributes in the console
    attrs = vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))

    # --- Return populated object from Parameters class
    return p
