'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_computeCompression     import Parameters
from shutil                                               import copyfile
import package_utils.fold_handler                         as fh
import os

def setParameters():

    p = Parameters(
        PSEG        = "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/tech_002/seg_gt",
        PRES        = "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/COMPRESSION",
        PCF         = "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/tech_002/seq/CF.txt",
        EXPERIMENT  = 'seg_gt',
        ROI         = (100, 150)
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

