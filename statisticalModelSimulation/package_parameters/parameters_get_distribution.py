import package_utils.fold_handler as fh
import os

# ----------------------------------------------------------------------------------------------------------------------
class Parameters:

    def __init__(
            self,
            PIMAGES,
            PBORDERS,
            PRES
    ):

        self.IMAGES = PIMAGES
        self.PBORDERS = PBORDERS
        self.PRES = PRES