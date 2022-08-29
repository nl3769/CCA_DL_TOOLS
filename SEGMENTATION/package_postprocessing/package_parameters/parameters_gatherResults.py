import package_utils.fold_handler as fh
import os

class Parameters:

    def __init__(
        self,
        PRES,
        PDATA
):

        self.PRES         = PRES
        self.PDATA        = PDATA

        #self.PPICKLE    = os.path.join(self.PRES, 'val_seg')

        # --- create directories
        #fh.create_dir(self.PPICKLE)

