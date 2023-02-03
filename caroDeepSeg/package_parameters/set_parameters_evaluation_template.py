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

    p=Parameters(
        PRES='/home/laine/Desktop/segmentation_results',
        PA1='/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Manual-A1',
        PA1BIS='/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Manual-A1s',
        PA2='/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Manual-A2',
        PCF='/run/media/laine/DISK/PROJECTS_IO/DATACUBS/CF',
        PMETHODS={
            "Computerized-UCY_CY": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-UCY_CY",
            "Computerized-CNR_I": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-CNR_IT",
            "Computerized-INESTEC_PT": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-INESCTEC_PT",
            "Computerized-POLITO_IT": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-POLITO_IT",
            "Computerized-POLITO_UNET": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-POLITO_UNET",
            "Computerized-TMU_DE": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-TUM_DE",
            "Computerized-CREATIS_TF": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_TF",
            "Computerized-CREATIS_PH_v0": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_PH_v0",
            "Computerized-CREATIS_PH_v1": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_PH_v1",
            "Computerized-CREATIS_PH_UNION_15_PXL": "/run/media/laine/DISK/PROJECTS_IO/DATACUBS/LIMA-Profiles-interpolated/Computerized-CREATIS_PH_UNION_15_PXL"
        },
        SET=['clin', 'tech', 'clin+tech']
    )

    pparam=os.path.join(p.PRES, 'backup_parameters')
    fh.create_dir(pparam)

    # --- Print all attributes in the console
    attrs=vars(p)
    print('----------------------------------------------------------------')
    print('\n'.join("%s: %s" % item for item in attrs.items()))
    print('----------------------------------------------------------------')

    # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
    copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_database.py'))

    # --- Return populated object from Parameters class
    return p

# ----------------------------------------------------------------------------------------------------------------------