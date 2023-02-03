'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os

from shutil import copyfile
from package_parameters.parameters_training_seg import Parameters
import package_utils.fold_handler as fh


# ----------------------------------------------------------------------------------------------------------------------
def setParameters():
    p = Parameters(
        PDATA='/run/media/laine/DISK/PROJECTS_IO/SEGMENTATION/REFERENCES',                                              # PATH TO LOAD DATA
        PRES='/run/media/laine/DISK/PROJECTS_IO/SEGMENTATION/TRAINING_RESULTS/SEG_TRAINING_FOLD_0',                     # PATH TO SAVE TRAINING RESULTS
        PSPLIT='/home/laine/Documents/PROJECTS_IO/CARODEEPSEG/SPLIT_PATIENT/fold_0',                                    # PATH TO LOAD TRAINING/VALIDATION/TESTING SUBSET
        IN_VIVO=True,                                                                                                   # REAL DATA
        LEARNING_RATE=0.01,                                                                                             # LEARNING RATE VALUE
        BATCH_SIZE=2,                                                                                                   # BATCH SIZE
        NB_EPOCH=50,                                                                                                    # TOTAL NUMBER OF EPOCH
        EARLY_STOP=10,                                                                                                  # EARLY STOP VALUE
        VALIDATION=True,                                                                                                # USE VALIDATION SUBSET
        DROPOUT=0.1,                                                                                                    # DROPOUT VALUE
        WORKERS=0,                                                                                                      # NUMBER OF WORKERS FOR THE DATA LOADER (SET TO 0 FOR DEBUG PURPOSE)
        USER='LAINE',                                                                                                   # USER NAME
        EXPNAME='test_early_stop',                                                                                      # NAME OF THE EXPERIMENT
        USE_WANDB=False,                                                                                                # USE OR NOT W&B
        ENTITY='nl37',                                                                                                  # NAME OF W&B USER
        DEVICE='cuda',                                                                                                  # SELECT DEVICE TO RUN (cuda or cpu)
        RESTORE_CHECKPOINT=True,                                                                                        # RESTORE MODEL (FOR FINE TUNING)
        # --- parameters for unet
        KERNEL_SIZE=(3, 3),                                                                                             # KERNEL SIZE OF THE MODEL
        PADDING=(1, 1),                                                                                                 # PADDING SIZE OF THE MODEL
        USE_BIAS=True,                                                                                                  # USE BIAS OR NOT
        NGF=32,                                                                                                         # NUMBER OF INPUT FEATURES OF THE UNET
        NB_LAYERS=4                                                                                                     # NUMBER OF LAYERS OF THE UNET
    )

    pparam = os.path.join(p.PRES, p.EXPNAME, 'parameters')
    fh.create_dir(p.PRES)
    fh.create_dir(pparam)
    # --- Print all attributes in the console
    attrs = vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')
    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)),
             os.path.join(pparam, 'get_parameters_training.py'))
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