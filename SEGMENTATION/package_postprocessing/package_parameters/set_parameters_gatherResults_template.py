'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from package_parameters.parameters_gatherResults        import Parameters
from shutil                                             import copyfile
import package_utils.fold_handler                       as fh
import os

def setParameters():

    p = Parameters(
        PRES              = "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/COMPRESSION/gather_results",
        PDATA             = {
            "seg_caroSegDeep-GAN-without_fine_tuning"       : "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/COMPRESSION/seg_caroSegDeep-GAN-without_fine_tuning",
            "seg_caroSegDeep-SIMULATED-without_fine_tuning" : "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/COMPRESSION/seg_caroSegDeep-SIMULATED-without_fine_tuning",
            "seg_gt"                                        : "/home/laine/Documents/PROJECTS_IO/SEGMENTATION/POST_PROCESSING/COMPRESSION/seg_gt"
            }
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

