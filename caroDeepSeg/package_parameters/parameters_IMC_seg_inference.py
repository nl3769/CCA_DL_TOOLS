import os.path

import package_utils.fold_handler as fh

# ----------------------------------------------------------------------------------------------------------------------
class Parameters:

    def __init__(self,
                 PDATA,
                 PSEG_REF,
                 PCF,
                 PMODELIMC,
                 PMODELFW,
                 PRES,
                 EXPNAME,
                 PATH_FW_REFERENCES,
                 PATIENT_NAME,
                 DEVICE,
                 FW_DETECTION,
                 ROI_WIDTH,
                 PIXEL_WIDTH,
                 PIXEL_HEIGHT,
                 SHIFT_X,
                 SHIFT_Z,
                 FW_INITIALIZATION,
                 KERNEL_SIZE,
                 PADDING,
                 USE_BIAS,
                 NGF,
                 NB_LAYERS):

        self.PDATA = PDATA
        self.PSEG_REF = PSEG_REF
        self.PCF = PCF
        self.PMODELIMC = PMODELIMC
        self.PMODELFW = PMODELFW
        self.PRES = PRES
        self.EXPNAME = EXPNAME
        self.PATH_FW_REFERENCES = PATH_FW_REFERENCES
        self.PATIENT_NAME = PATIENT_NAME
        self.DEVICE = DEVICE
        self.FW_DETECTION = FW_DETECTION
        self.ROI_WIDTH = ROI_WIDTH
        self.PIXEL_WIDTH = PIXEL_WIDTH
        self.PIXEL_HEIGHT = PIXEL_HEIGHT
        self.SHIFT_X = SHIFT_X
        self.SHIFT_Z = SHIFT_Z
        self.FW_INITIALIZATION = FW_INITIALIZATION
        self.KERNEL_SIZE = KERNEL_SIZE
        self.PADDING = PADDING
        self.USE_BIAS = USE_BIAS
        self.NGF = NGF
        self.NB_LAYERS = NB_LAYERS

        # --- path to save results
        self.PATH_SEG_VISUAL = os.path.join(PRES, EXPNAME, "SEG_VISUAL")
        self.PATH_EXEC_TIME = os.path.join(PRES, EXPNAME, "EXEC_TIME")
        self.PATH_NB_PATCHES = os.path.join(PRES, EXPNAME, "NB_PATCHES")
        self.PATH_SEGMENTATION_RESULTS = os.path.join(PRES, EXPNAME, "SEGMENTATION_RESULTS")

        # --- create directories
        fh.create_dir(self.PRES)
        fh.create_dir(self.PATH_SEG_VISUAL)
        fh.create_dir(self.PATH_EXEC_TIME)
        fh.create_dir(self.PATH_NB_PATCHES)
        fh.create_dir(self.PATH_SEGMENTATION_RESULTS)