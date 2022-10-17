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
        MODEL_NAME                  = 'gma',                                                                           # chose gma or raft
        PDATA                       = '/home/laine/PROJECTS_IO/DATA/OPTICAL_FLOW/FlyingChairs/data',            # PATH TO LOAD DATA
        PRES                        = '/home/laine/PROJECTS_IO/CARODEEPFLOW/GMA_flyingChairs_pretraining_01',          # PATH TO SAVE TRAINING RESULTS
        PSPLIT                      = '/home/laine/PROJECTS_IO/DATA/OPTICAL_FLOW/FlyingChairs/data',
        LEARNING_RATE               = 0.0001,
        BATCH_SIZE                  = 8,                                                                        # size of a batch
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
        EXPNAME                     = 'GMA_PRETRAINED_FLYINGCHAIR',
        DEVICE                      = 'cuda',                                                                   # cuda/cpu
        RESTORE_CHECKPOINT          = True,
        FEATURES                    = 'split',                                                                 # shared or split
        # --- If feature is split, then chose parameters for Unet
        KERNEL_SIZE                 = (3, 3),
        PADDING                     = (1, 1),
        USE_BIAS                    = True,
        NGF                         = 32,  # number of input features of the Unet
        NB_LAYERS                   = 4)

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
