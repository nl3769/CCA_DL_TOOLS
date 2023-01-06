import os
from package_utils.utils                import check_dir

class Parameters:

    def __init__(
        self,
        MODEL_NAME,
        PDATA,
        DATABASE,
        VALIDATION,
        RESTORE_CHECKPOINT,
        LOSS,
        lambda_GAN,
        lambda_pixel,
        LEARNING_RATE,
        BATCH_SIZE,
        NB_EPOCH,
        IMAGE_NORMALIZATION,
        DATA_AUG,
        KERNEL_SIZE,
        PADDING,
        USE_BIAS,
        UPCONV,
        NGF,
        NB_LAYERS,
        IMG_SIZE,
        DROPOUT,
        WORKERS,
        EARLY_STOP,
        OUTPUT_ACTIVATION,
        PATH_RES,
        USE_WB):

        self.MODEL_NAME = MODEL_NAME
        self.PDATA = PDATA
        self.DATABASE = DATABASE
        self.VALIDATION = VALIDATION
        self.RESTORE_CHECKPOINT = RESTORE_CHECKPOINT
        self.LOSS = LOSS
        self.lambda_GAN = lambda_GAN
        self.lambda_pixel = lambda_pixel
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.NB_EPOCH = NB_EPOCH
        self.IMAGE_NORMALIZATION = IMAGE_NORMALIZATION
        self.DATA_AUG = DATA_AUG
        self.KERNEL_SIZE = KERNEL_SIZE
        self.PADDING = PADDING
        self.USE_BIAS = USE_BIAS
        self.UPCONV = UPCONV
        self.NGF = NGF
        self.NB_LAYERS = NB_LAYERS
        self.IMG_SIZE = IMG_SIZE
        self.DROPOUT = DROPOUT
        self.WORKERS = WORKERS
        self.OUTPUT_ACTIVATION = OUTPUT_ACTIVATION
        self.EARLY_STOP = EARLY_STOP
        self.PATH_RES = PATH_RES
        self.USE_WB = USE_WB
        self.PATH_RANDOM_PRED_TRN = os.path.join(self.PATH_RES, 'training_pred')
        self.PATH_PRINT_MODEL = os.path.join(self.PATH_RES, 'print_model')
        self.PATH_SAVE_MODEL = os.path.join(self.PATH_RES, 'model')
        self.PATH_MODEL_HISTORY = os.path.join(self.PATH_RES, 'training_history')
        self.PATH_SAVE_FIGURE = os.path.join(self.PATH_RES, 'training_figure')
        self.PATH_PRED_EVALUATION = os.path.join(self.PATH_RES, 'evaluation_pred')
        self.PATH_SAVE_CVS = os.path.join(self.PATH_RES, 'evaluation_CVS')

        # --- create directories
        check_dir(self.PATH_RANDOM_PRED_TRN)
        check_dir(self.PATH_PRINT_MODEL)
        check_dir(self.PATH_SAVE_MODEL)
        check_dir(self.PATH_MODEL_HISTORY)
        check_dir(self.PATH_SAVE_FIGURE)
        check_dir(self.PATH_PRED_EVALUATION)
        check_dir(self.PATH_SAVE_CVS)
