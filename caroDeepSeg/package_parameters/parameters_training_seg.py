import package_utils.fold_handler as fh
import os

class Parameters:

    def __init__(
        self,
        PDATA,
        PRES,
        PSPLIT,
        IN_VIVO,
        LEARNING_RATE,
        BATCH_SIZE,
        NB_EPOCH,
        VALIDATION,
        DROPOUT,
        WORKERS,
        USER,
        EXPNAME,
        DEVICE,
        RESTORE_CHECKPOINT,
        ENTITY,
        KERNEL_SIZE,
        PADDING,
        USE_BIAS,
        NGF,
        NB_LAYERS
    ):

        self.PDATA                              = PDATA
        self.PRES                               = PRES
        self.PSPLIT                             = PSPLIT
        self.IN_VIVO                            = IN_VIVO
        self.LEARNING_RATE                      = LEARNING_RATE
        self.BATCH_SIZE                         = BATCH_SIZE
        self.NB_EPOCH                           = NB_EPOCH
        self.VALIDATION                         = VALIDATION
        self.DROPOUT                            = DROPOUT
        self.WORKERS                            = WORKERS
        self.USER                               = USER
        self.EXPNAME                            = EXPNAME
        self.DEVICE                             = DEVICE
        self.RESTORE_CHECKPOINT                 = RESTORE_CHECKPOINT
        self.ENTITY                             = ENTITY
        self.KERNEL_SIZE                        = KERNEL_SIZE
        self.PADDING                            = PADDING
        self.USE_BIAS                           = USE_BIAS
        self.NGF                                = NGF
        self.NB_LAYERS                          = NB_LAYERS
        self.PATH_RANDOM_PRED_TRN               = os.path.join(self.PRES, self.EXPNAME, 'training_pred')
        self.PATH_PRINT_MODEL                   = os.path.join(self.PRES, self.EXPNAME, 'print_model')
        self.PATH_SAVE_MODEL                    = os.path.join(self.PRES, self.EXPNAME, 'saved_model')
        self.PATH_MODEL_HISTORY                 = os.path.join(self.PRES, self.EXPNAME, 'training_history')
        self.PATH_SAVE_FIGURE                   = os.path.join(self.PRES, self.EXPNAME, 'training_figure')
        self.PATH_TST                           = os.path.join(self.PRES, self.EXPNAME, 'test_results')
        self.PATH_WANDB                         = os.path.join(self.PRES, self.EXPNAME)

        # --- create directories
        fh.create_dir(self.PATH_RANDOM_PRED_TRN)
        fh.create_dir(self.PATH_PRINT_MODEL)
        fh.create_dir(self.PATH_SAVE_MODEL)
        fh.create_dir(self.PATH_MODEL_HISTORY)
        fh.create_dir(self.PATH_SAVE_FIGURE)
        fh.create_dir(self.PATH_TST)
        fh.create_dir(self.PATH_WANDB)