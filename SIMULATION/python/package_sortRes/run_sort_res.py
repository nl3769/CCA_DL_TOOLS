import os
import shutil
import glob
from icecream import ic


# ----------------------------------------------------------------------------------------------------------------------
class simulationHandler():

    def __init__(self, pres, pdata):

        self.pres           = pres
        self.pdata          = pdata
        self.path           = {}
        self.success        = {}
        self.tx_events      = 128
        self.tx_store       = "raw_data/raw_"
        self.bmode_store    = "bmode_result/RF"

    # ------------------------------------------------------------------------------------------------------------------
    def get_pres(self):

        patients = os.listdir(self.pdata)
        patients.sort()
        for patient in patients:
            self.path[patient] = {}
            id_seq = os.listdir(os.path.join(self.pdata, patient))
            id_seq.sort()
            for seq in id_seq:
                self.path[patient][seq] = os.path.join(self.pdata, patient, seq)

    # ------------------------------------------------------------------------------------------------------------------
    def get_success(self):

        for patient in self.path.keys():
            self.success[patient] = {}
            for seq in self.path[patient]:
                pres = self.path[patient][seq]
                condition = self.check_tx_success(pres, self.tx_store, self.tx_events)
                if condition:
                    condition = self.check_bmode_success(pres, self.bmode_store)

                self.success[patient][seq] = condition

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def check_bmode_success(path, substr):
        """ Check if bmode image is created. """

        if os.path.isdir(os.path.join(path, substr)):
            files = glob.glob(os.path.join(path, substr, "*" + "_bmode.png" + "*"))

            if len(files) == 1:
                condition = True
            else:
                condition = False
        else:
            condition = False

        return condition

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def check_tx_success(path, substr, tx_events):
        """ Check if all tx events are successful. """

        raw_files = os.listdir(os.path.join(path, substr))
        raw_files.sort()
        if len(raw_files) == tx_events:
            condition = True
        else:
            condition = False

        return condition

    # ------------------------------------------------------------------------------------------------------------------
    def move_success(self):
        for patient in self.path.keys():
            for id_seq in self.path[patient].keys():
                source_path = self.path[patient][id_seq]
                if self.success[patient][id_seq]:
                    dest_path = os.path.join(self.pres, patient)
                    create_dir(dest_path)
                    ic(source_path)
                    ic(dest_path)
                    shutil.move(source_path, dest_path)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_pres()
        self.get_success()
        self.move_success()

# ----------------------------------------------------------------------------------------------------------------------
def create_dir(path):

    try:
        os.makedirs(path)
    except OSError as error:
        print(error)

# ----------------------------------------------------------------------------------------------------------------------
def main():

    #pres  = '/run/media/laine/HDD/PROJECTS_IO/SIMULATION/CUBS'
    pres  = '/run/media/laine/HDD/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER'
    pdata = '/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER'

    simHandler = simulationHandler(pres, pdata)
    simHandler()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ Sort successful simulation. """

    main()
