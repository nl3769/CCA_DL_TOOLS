import package_utils.fold_handler as fh
import os

# ----------------------------------------------------------------------------------------------------------------------
class Parameters:

    def __init__(
            self,
            PIMAGES,
            PCF,
            PLUMEN,
            PINTERFACES,
            PRES
    ):

        self.PIMAGES = PIMAGES
        self.PCF = PCF
        self.PLUMEN = PLUMEN
        self.PINTERFACES = PINTERFACES
        self.PRES = PRES

        self.PRES_IMAGES = os.path.join(self.PRES, "images")
        self.PRES_STAT_MODEL = os.path.join(self.PRES, "stat_model")

        # --- create diretories
        fh.create_dir(self.PRES)
        fh.create_dir(self.PRES_IMAGES)
        fh.create_dir(self.PRES_STAT_MODEL)