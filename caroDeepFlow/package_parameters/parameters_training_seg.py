
class Parameters:

    def __init__(self,
                 PDATA,
                 PRES,
                 PSPLIT,
                 LEARNING_RATE,
                 EPOCH,
                 BATCH_SIZE,  # size of a batch
                 NB_EPOCH,
                 VALIDATION,
                 DROPOUT,  # dropout during training
                 WORKERS,
                 USER,
                 EXPNAME,
                 DEVICE,  # cuda/cpu
                 RESTORE_CHECKPOINT
                 ):

            self.PDATA = PDATA
            self.PRES = PRES
            self.PSPLIT = PSPLIT
            self.LEARNING_RATE = LEARNING_RATE
            self.EPOCH = EPOCH
            self.BATCH_SIZE = BATCH_SIZE
            self.NB_EPOCH = NB_EPOCH
            self.VALIDATION = VALIDATION
            self.DROPOUT = DROPOUT
            self.WORKERS = WORKERS
            self.USER = USER
            self.EXPNAME = EXPNAME
            self.DEVICE = DEVICE
            self.RESTORE_CHECKPOINT = RESTORE_CHECKPOINT