import package_utils.fold_handler as fh

class Parameters:
    def __init__(self,
                 PDATA,
                 PRES,
                 ROI_WIDTH,
                 PIXEL_WIDTH,
                 PIXEL_HEIGHT,
                 SHIFT_X,
                 SHIFT_Z
                 ):

        self.PDATA = PDATA
        self.PRES = PRES
        self.ROI_WIDTH = ROI_WIDTH
        self.PIXEL_WIDTH = PIXEL_WIDTH
        self.PIXEL_HEIGHT = PIXEL_HEIGHT
        self.SHIFT_X = SHIFT_X
        self.SHIFT_Z = SHIFT_Z