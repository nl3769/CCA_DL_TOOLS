import os

from package_dataloader. diffDataloader             import diffDataloader
from glob                                           import glob

class dataHandler(diffDataloader):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p, data_aug, set: str):
        super(dataHandler, self).__init__(p, data_aug)

        with open(p.DATABASE[set], 'r') as f:
            root = f.readlines()
        root = [os.path.join(p.PDATA, path.replace('\n', '')) for path in root]

        for fname in root:

            # --- path to images
            org = sorted(glob(os.path.join(fname, '*org.png')))[0]
            self.org_list += [[org]]

            # --- path to flow
            sim = sorted(glob(os.path.join(fname, '*simulated.png')))[0]
            self.sim_list += [[sim]]

        # self.sim_list = self.sim_list[:20]
        # self.org_list = self.org_list[:20]

    # ------------------------------------------------------------------------------------------------------------------

def get_last_element(in_str: str):

    return in_str.split('/')[-1]
