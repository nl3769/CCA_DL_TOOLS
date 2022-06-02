
class Parameters:

    def __init__(self,
                 PDATA,
                 PRES,
                 PSPLIT,
                 LEARNING_RATE,
                 EPOCH,
                 BATCH_SIZE,
                 NB_EPOCH,
                 VALIDATION,
                 MIXED_PRECISION,
                 IMG_SIZE,
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
                 USER,
                 EXPNAME
):

        self.PDATA = PDATA
        self.PRES = PRES
        self.PSPLIT = PSPLIT
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.NB_EPOCH = NB_EPOCH
        self.VALIDATION = VALIDATION
        self.MIXED_PRECISION = MIXED_PRECISION
        self.IMG_SIZE = IMG_SIZE
        self.DROPOUT = DROPOUT
        self.GAMMA = GAMMA
        self.ADD_NOISE = ADD_NOISE
        self.CORRELATION_LEVEL = CORRELATION_LEVEL
        self.CORRELATION_RADIUS = CORRELATION_RADIUS
        self.NB_ITERATION = NB_ITERATION
        self.ALTERNATE_COORDINATE = ALTERNATE_COORDINATE
        self.WORKERS = WORKERS
        self.POSITION_ONLY = POSITION_ONLY
        self.POSITION_AND_CONTENT = POSITION_AND_CONTENT,
        self.NUM_HEAD = NUM_HEAD
        self.USER = USER
        self.EXPNAME = EXPNAME
