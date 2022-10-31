import package_utils.fold_handler as fh
import os

class Parameters:

    def __init__(
        self,
        MODEL_NAME,
        PDATA,
        PRES,
        PSPLIT,
        LEARNING_RATE,
        BATCH_SIZE,
        NB_EPOCH,
        VALIDATION,
        DROPOUT,
        GAMMA,
        ADD_NOISE,
        CORRELATION_LEVEL,
        CORRELATION_RADIUS,
        NB_ITERATION,
        ALTERNATE_COORDINATE,
        WORKERS,
        POSITION_ONLY,
        POSITION_AND_CONTENT,
        NUM_HEAD,
        ADVENTICIA_DIM,
        USER,
        EXPNAME,
        DEVICE,
        RESTORE_CHECKPOINT,
        SYNTHETIC_DATASET,
        KERNEL_SIZE,
        PADDING,
        USE_BIAS,
        NGF,
        NB_LAYERS
):
        
        self.MODEL_NAME                         = MODEL_NAME
        self.PDATA                              = PDATA
        self.PRES                               = PRES
        self.PSPLIT                             = PSPLIT
        self.LEARNING_RATE                      = LEARNING_RATE
        self.BATCH_SIZE                         = BATCH_SIZE
        self.NB_EPOCH                           = NB_EPOCH
        self.VALIDATION                         = VALIDATION
        self.DROPOUT                            = DROPOUT
        self.GAMMA                              = GAMMA
        self.ADD_NOISE                          = ADD_NOISE
        self.CORRELATION_LEVEL                  = CORRELATION_LEVEL
        self.CORRELATION_RADIUS                 = CORRELATION_RADIUS
        self.NB_ITERATION                       = NB_ITERATION
        self.ALTERNATE_COORDINATE               = ALTERNATE_COORDINATE
        self.WORKERS                            = WORKERS
        self.POSITION_ONLY                      = POSITION_ONLY
        self.POSITION_AND_CONTENT               = POSITION_AND_CONTENT
        self.NUM_HEAD                           = NUM_HEAD
        self.ADVENTICIA_DIM                     = ADVENTICIA_DIM
        self.USER                               = USER
        self.EXPNAME                            = EXPNAME
        self.DEVICE                             = DEVICE
        self.RESTORE_CHECKPOINT                 = RESTORE_CHECKPOINT
        self.SYNTHETIC_DATASET                  = SYNTHETIC_DATASET
        self.KERNEL_SIZE                        = KERNEL_SIZE
        self.PADDING                            = PADDING
        self.USE_BIAS                           = USE_BIAS
        self.NGF                                = NGF
        self.NB_LAYERS                          = NB_LAYERS
        self.PATH_RANDOM_PRED_TRN               = os.path.join(self.PRES, 'training_pred')
        self.PATH_PRINT_MODEL                   = os.path.join(self.PRES, 'print_model')
        self.PATH_SAVE_MODEL                    = os.path.join(self.PRES, 'model')
        self.PATH_MODEL_HISTORY                 = os.path.join(self.PRES, 'training_history')
        self.PATH_SAVE_FIGURE                   = os.path.join(self.PRES, 'training_figure')
        self.PATH_SAVE_PRED_TRAINING            = os.path.join(self.PRES, 'training_pred')

        # --- create directories
        fh.create_dir(self.PATH_RANDOM_PRED_TRN)
        fh.create_dir(self.PATH_PRINT_MODEL)
        fh.create_dir(self.PATH_SAVE_MODEL)
        fh.create_dir(self.PATH_MODEL_HISTORY)
        fh.create_dir(self.PATH_SAVE_FIGURE)
        fh.create_dir(self.PATH_SAVE_PRED_TRAINING)
