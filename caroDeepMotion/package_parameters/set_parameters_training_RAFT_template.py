'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from shutil                                         import copyfile
from package_parameters.parameters_training_RAFT    import Parameters
import package_utils.fold_handler                   as fh
import os

def setParameters():

    p  =  Parameters(
        MODEL_NAME = 'raft',                                                                                              # chose gma or raft
        PDATA = '/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/database_training_IMAGENET',                # PATH TO LOAD DATA
        PRES = '/home/laine/Desktop/trn_debug',                                                                           # PATH TO SAVE TRAINING RESULTS
        PSPLIT = '/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/SPLIDATA',
        MAGNITUDE_MOTION = None,
        LEARNING_RATE = 0.0001,
        BATCH_SIZE = 2,                                                                                                   # size of a batch
        NB_EPOCH = 150,                                                                                                   # number rof epochs
        VALIDATION = True,                                                                                                # if validation set exists
        DROPOUT = 0.0,                                                                                                    # dropout during training
        GAMMA = 0.8,                                                                                                      # gamma coefficient for the loss function
        ADD_NOISE = False,                                                                                                # add noise during training (it is not implemented yet)
        CORRELATION_LEVEL = 4,
        CORRELATION_RADIUS = 4,                                                                                           # size of the neighbours for the correlation
        NB_ITERATION = 12,                                                                                                # number of iteration to refine the estimated flow
        WORKERS = 0,                                                                                                      # number of workers during training
        ADVENTICIA_DIM = 1,                                                                                               # part of adventitia in mm
        ENTITY = 'nl37',                                                                                                  # name of the user for W&B
        EXPNAME = 'TEST_TRAINING',                                                                                       # name fo the exeperiment
        DEVICE = 'cuda',                                                                                                  # cuda/cpu
        RESTORE_CHECKPOINT = True,                                                                                        # for fine tuning or continue training
        SYNTHETIC_DATASET = True,                                                                                         # train on generated database (True)
        USE_WANDB = False,                                                                                                # use W&B
        KERNEL_SIZE = (3, 3),                                                                                             # kernel size
        PADDING = (1, 1),                                                                                                 # padding size
        USE_BIAS = True)                                                                                                  # use bias

    pparam  =  os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)
    # --- Print all attributes in the console
    attrs  =  vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')
    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))
    # --- Return populated object from Parameters class
    return p
