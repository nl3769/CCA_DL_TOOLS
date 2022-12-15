import os
import package_utils.fold_handler as fh

class Parameters:
    def __init__(
        self,
        PRES,
        PA1,
        PA1BIS,
        PA2,
        PCF,
        PMETHODS,
        SET
        ):

        self.PRES = PRES
        self.PA1 = PA1
        self.PA1BIS = PA1BIS
        self.PA2 = PA2
        self.PCF = PCF
        self.PMETHODS = PMETHODS
        self.SET = SET

        self.PRESCSV = os.path.join(self.PRES, 'CVS')
        self.PLOT = os.path.join(self.PRES, 'PLOT')
        self.PUNPROCESSED = os.path.join(self.PRES, 'UNPROCESSED_IMAGES')

        # --- create directory
        fh.create_dir(PRES)
        fh.create_dir(self.PRESCSV)
        fh.create_dir(self.PLOT)
        fh.create_dir(self.PUNPROCESSED)