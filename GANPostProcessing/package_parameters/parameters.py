class Parameters:

    def __init__(self,
                 MODEL_NAME,
                 PDATA,
                 DATABASE,
                 VALIDATION,
                 LOSS,
                 LOSS_BALANCE,
                 CASCADE_FILTERS,
                 LEARNING_RATE,
                 BATCH_SIZE,
                 NB_EPOCH,
                 NORMALIZATION,
                 IMAGE_NORMALIZATION,
                 KERNEL_SIZE,
                 BILINEAR,
                 IMG_SIZE,
                 DROPOUT,
                 WORKERS,
                 EARLY_STOP,
                 PATH_RES):

        self.MODEL_NAME             = MODEL_NAME
        self.PDATA                  = PDATA
        self.DATABASE               = DATABASE
        self.VALIDATION             = VALIDATION
        self.LOSS                   = LOSS
        self.LOSS_BALANCE           = LOSS_BALANCE
        self.CASCADE_FILTERS        = CASCADE_FILTERS
        self.LEARNING_RATE          = LEARNING_RATE
        self.BATCH_SIZE             = BATCH_SIZE
        self.NB_EPOCH               = NB_EPOCH
        self.NORMALIZATION          = NORMALIZATION
        self.IMAGE_NORMALIZATION    = IMAGE_NORMALIZATION
        self.KERNEL_SIZE            = KERNEL_SIZE
        self.BILINEAR               = BILINEAR
        self.IMG_SIZE               = IMG_SIZE
        self.DROPOUT                = DROPOUT
        self.WORKERS                = WORKERS
        self.EARLY_STOP             = EARLY_STOP
        self.PATH_RES               = PATH_RES