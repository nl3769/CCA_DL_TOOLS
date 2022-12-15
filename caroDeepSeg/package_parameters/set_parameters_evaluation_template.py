'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from shutil                                             import copyfile
from package_parameters.parameters_evaluation           import Parameters
import os
import package_utils.fold_handler                       as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

    p = Parameters(
        PRES = '/home/laine/Desktop/segmentation_results',
        PA1 = '/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Manual-A1',
        PA1BIS = '/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Manual-A1s',
        PA2 =  '/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Manual-A2',
        PCF = '/home/laine/Documents/PROJECTS_IO/DATA/CUBS/CF',
        PMETHODS = {
            "Computerized-UCY_CY": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-UCY_CY",
            "Computerized-CNR_I": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-CNR_IT",
            "Computerized-INESTEC_PT": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-INESCTEC_PT",
            "Computerized-POLITO_IT": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-POLITO_IT",
            "Computerized-POLITO_UNET": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-POLITO_UNET",
            "Computerized-TMU_DE": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-TUM_DE",
            "Computerized-CREATIS_TF": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_TF",
            "Computerized-CREATIS_PH_v0": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_PH_v0",
            "Computerized-CREATIS_PH_v1": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_PH_v1",
            "Computerized-CREATIS_PH_UNION_15_PXL": "/home/laine/Documents/PROJECTS_IO/DATA/CUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_PH_UNION_15_PXL"
        },
        SET = 'clin+tech' # chose 'clin', 'tech' or 'clin+tech'
    )

    pparam = os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)

    # --- Print all attributes in the console
    attrs = vars(p)
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_database.py'))

    # --- Return populated object from Parameters class
    return p

# ----------------------------------------------------------------------------------------------------------------------