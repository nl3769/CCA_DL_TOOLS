import os

import package_utils.fold_handler                   as fh
from package_evaluation.evaluationHandler           import evaluationHandler

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class Parameters:
    def __init__(
        self,
        PRES,
        PA1,
        PA1BIS,
        PA2,
        PCF,
        PMETHODS,
        ):

        self.PRES = PRES
        self.PA1 = PA1
        self.PA1BIS = PA1BIS
        self.PA2 = PA2
        self.PCF = PCF
        self.PMETHODS = PMETHODS

        # --- create directory
        fh.create_dir(PRES)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def main():
   
    parameters = Parameters(
        PRES = '/home/laine/Desktop/segmentation_results',
        PA1 = '/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Manual-A1',
        PA1BIS = '/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Manual-A1s',
        PA2 =  '/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Manual-A2',
        PCF = '/home/laine/Desktop/data_cubs/CF',
        PMETHODS = {
            "Computerized-CNR_I": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-CNR_IT",
            "Computerized-CREATIS": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-CREATIS",
            "Computerized-INESTEC_PT": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-INESCTEC_PT",
            "Computerized-POLITO_IT": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-POLITO_IT",
            "Computerized-POLITO_UNET": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-POLITO_UNET",
            "Computerized-TMU_DE": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-TUM_DE",
            "Computerized-UCY_CY": "/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Computerized-UCY_CY"}
    )
    eval = evaluationHandler(parameters)

    keys = ['A1bis', 'A2'] + list(eval.annotation_methods.keys())

    for key in keys:
        print(key)
        # key = 'Computerized-INESTEC_PT'
        eval.get_diff(key)
        eval.get_MAE(key)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
