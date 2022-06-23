import os
import math

from icecream                           import ic
from package_dataset.splitData          import splitData

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pres        = '/home/laine/cluster/PROJECTS_IO/DATA/GAN'
    subdataset  = ['PATHO_ANDRE_25', 'MEIBURGER_1_FRAME', 'HEALTHY_ANDRE_57']
    psave       = '/home/laine/cluster/PROJECTS_IO/DATA/GAN/split_v01'

    info ={'pres'           : pres,
           'subdataset'     : subdataset,
           'training_size'  : 70,
           'validation_size': 10,
           'psave'          : psave}
    ic(info)


    splitClass = splitData(info)
    splitClass.split_data()
    splitClass.save_res()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()