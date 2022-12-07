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

        # --- create directory
        fh.create_dir(PRES)