import package_utils.fold_handler as fh

class Parameters:

    def __init__(
        self,
        MODEL_NAME,
        PRES,
        PSAVE,
        PDATA,
        DROPOUT,
        CORRELATION_LEVEL,
        CORRELATION_RADIUS,
        RESTORE_CHECKPOINT,
        ALTERNATE_COORDINATE,
        PSPLIT,
        PIXEL_WIDTH,
        PIXEL_HEIGHT,
        ROI_WIDTH,
        SHIFT_X,
        SHIFT_Z):

        self.MODEL_NAME = MODEL_NAME
        self.PRES = PRES
        self.PSAVE= PSAVE
        self.PDATA = PDATA
        self.DROPOUT = DROPOUT
        self.CORRELATION_LEVEL = CORRELATION_LEVEL
        self.CORRELATION_RADIUS = CORRELATION_RADIUS
        self.RESTORE_CHECKPOINT = RESTORE_CHECKPOINT
        self.ALTERNATE_COORDINATE = ALTERNATE_COORDINATE
        self.PSPLIT = PSPLIT
        self.PIXEL_WIDTH = PIXEL_WIDTH
        self.PIXEL_HEIGHT = PIXEL_HEIGHT
        self.ROI_WIDTH = ROI_WIDTH
        self.SHIFT_X = SHIFT_X
        self.SHIFT_Z = SHIFT_Z