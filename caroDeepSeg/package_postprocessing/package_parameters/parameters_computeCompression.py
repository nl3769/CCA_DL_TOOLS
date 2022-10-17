import package_utils.fold_handler as fh
import os

class Parameters:

    def __init__(
        self,
        PSEG,
        PRES,
        PCF,
        EXPERIMENT,
        ROI
):

        self.PSEG       = PSEG
        self.PRES       = os.path.join(PRES, EXPERIMENT)
        self.PCF        = PCF
        self.ROI        = ROI

        self.PFIGURE    = os.path.join(self.PRES, 'figure')
        self.PBORDERS   = os.path.join(self.PRES, 'borders_info')
        self.PTEXT      = os.path.join(self.PRES, 'txt_info')
        self.PPICKLE    = os.path.join(self.PRES, 'val_seg')

        # --- create directories
        fh.create_dir(self.PFIGURE)
        fh.create_dir(self.PBORDERS)
        fh.create_dir(self.PTEXT)
        fh.create_dir(self.PPICKLE)

