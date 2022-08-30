import os
import math

from icecream                           import ic
from package_dataset.splitData          import splitData

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pres        = '/home/laine/cluster/PROJECTS_IO/DATA/GAN'
    subdataset  = ['GZ/PATHO_ANDRE_25', 'GZ/HEALTHY_ANDRE_57', 'MEIBURGER_1_FRAME', 'MCMASTER/CAMO_study', 'MCMASTER/CAMS_study', 'MCMASTER/RAM_study', 'MCMASTER/SWOLL_study', 'MCMASTER/TR_study']

    psave       = '/home/laine/cluster/PROJECTS_IO/DATA/GAN/split_00'

    info ={'pres'           : pres,
           'subdataset'     : subdataset,
           'training_size'  : 80,
           'validation_size': 10,
           'psave'          : psave}
    ic(info)


    splitClass = splitData(info)
    splitClass.split_data()
    splitClass.save_res()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
