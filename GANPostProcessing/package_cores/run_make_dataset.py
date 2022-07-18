import os
from icecream                           import ic
from package_dataset.dataSetBuilder     import dataSetBuilder

# ----------------------------------------------------------------------------------------------------------------------
def main():

    info = {'pdata': '/run/media/laine/HDD/PROJECTS_IO/SIMULATION/MEIBURGER_1_FRAME',
            'pres': '/home/laine/cluster/PROJECTS_IO/DATA/GAN/MEIBURGER_1_FRAME',
            'dataset': 'CUBS2',
            'subDataset': '',
            'pbmode': 'bmode_result/RF',
            'phantom': 'phantom'
            }

    ic(info)
    dataset = dataSetBuilder(info)
    dataset.get_fname()
    dataset.extract_images()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()