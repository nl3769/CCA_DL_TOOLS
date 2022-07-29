import os
import package_utils.fold_handler as fh

# -----------------------------------------------------------------------------------------------------------------------
class Parameters:

    def __init__(
        self,
        PDATA,
        DATABASE,
        VALIDATION,
        PRES,
        IMG_SIZE,
        IMAGE_NORMALIZATION,
        NB_LAYERS,
        NGF,
        KERNEL_SIZE,
        PADDING,
        USE_BIAS,
        INPUT_NC,
        OUTPUT_NC,
        TIME_EMB_DIM,
        TIME_STEP,
        NB_EPOCH,
        LEARNING_RATE,
        BATCH_SIZE,
        WORKERS,
        EARLY_STOP,
        BETA
        ):

        self.PDATA                          = PDATA
        self.DATABASE                       = DATABASE
        self.VALIDATION                     = VALIDATION
        self.PRES                           = PRES
        self.IMG_SIZE                       = IMG_SIZE
        self.IMAGE_NORMALIZATION            = IMAGE_NORMALIZATION
        self.NB_LAYERS                      = NB_LAYERS
        self.NGF                            = NGF
        self.KERNEL_SIZE                    = KERNEL_SIZE
        self.PADDING                        = PADDING
        self.USE_BIAS                       = USE_BIAS
        self.INPUT_NC                       = INPUT_NC
        self.OUTPUT_NC                      = OUTPUT_NC
        self.TIME_EMB_DIM                   = TIME_EMB_DIM
        self.TIME_STEP                      = TIME_STEP
        self.NB_EPOCH                       = NB_EPOCH
        self.LEARNING_RATE                  = LEARNING_RATE
        self.BATCH_SIZE                     = BATCH_SIZE
        self.WORKERS                        = WORKERS
        self.EARLY_STOP                     = EARLY_STOP
        self.BETA                           = BETA
        # --- path to store results
        self.PATH_SAVE_MODEL                = os.path.join(PRES, "models")
        self.PATH_MODEL_HISTORY             = os.path.join(PRES, "history")
        self.PATH_SAVE_FIGURE               = os.path.join(PRES, "trn_figures")
        self.PATH_RDM_PRED_DIFFUSION        = os.path.join(PRES, "trn_rdm_diffusion")
        self.PATH_RDM_PRED_FINAL_RES        = os.path.join(PRES, "trn_rdm_final_res")

        # --- create directories
        fh.create_dir(self.PATH_SAVE_MODEL)
        fh.create_dir(self.PATH_MODEL_HISTORY)
        fh.create_dir(self.PATH_SAVE_FIGURE)
        fh.create_dir(self.PATH_RDM_PRED_DIFFUSION)
        fh.create_dir(self.PATH_RDM_PRED_FINAL_RES)