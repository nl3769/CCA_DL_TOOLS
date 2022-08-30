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
        PSEG        = "/home/laine/cluster/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/tech_002/TECH_002/seg_caroSegDeep-GAN-with_fine_tuning",
        PRES        = "/home/laine/cluster/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/tech_002/COMPRESSION",
        PCF         = "/home/laine/cluster/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/tech_002/TECH_002/seq/CF.txt",
        EXPERIMENT  = 'seg_caroSegDeep-GAN-with_fine_tuning',
        ROI         = (110, 150)
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

