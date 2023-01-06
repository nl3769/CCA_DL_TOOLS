'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_fullImgMotion import Parameters
import package_utils.fold_handler as fh
import os
from shutil import copyfile

def setParameters():

    p=Parameters(
        MODEL_NAME='raft',
        PRES='/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/NETWORK_TRAINING/RAFT_PRETRAINED_FLYINGCHAIR_10_PX_FINE_TUNING',
        PSAVE='/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/PREDICTION/DL_METHOD_FULL_IMAGES',
        PDATA='/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/prepared_data_IMAGENET',
        DROPOUT=0,
        CORRELATION_LEVEL=4,
        CORRELATION_RADIUS=4,
        RESTORE_CHECKPOINT=True,
        ALTERNATE_COORDINATE=False,
        PSPLIT="/run/media/laine/DISK/PROJECTS_IO/caroSegMotion/IN_SILICO/REAL_DATA/SPLIDATA/validation_patients.txt",
        PIXEL_WIDTH=256,
        PIXEL_HEIGHT=256,
        ROI_WIDTH=5e-3,
        SHIFT_X=32,
        SHIFT_Z=32)

    pparam=os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)

    # --- Print all attributes in the console
    attrs=vars(p)
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))

    # --- Return populated object from Parameters class
    return p
