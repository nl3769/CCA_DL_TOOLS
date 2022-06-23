import os
from icecream                           import ic
from package_dataset.dataSetBuilder     import dataSetBuilder

# ----------------------------------------------------------------------------------------------------------------------
def main():

    info = {'pdata': '/package_core/media/laine/HDD/PROJECTS_IO/SIMULATION/PATHO_ANDRE_25',
            'pres': '/home/laine/cluster/PROJECTS_IO/DATA/GAN/PATHO_ANDRE_25',
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