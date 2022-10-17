'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import argparse
import importlib
import os
from package_database.databaseCUBSHandler       import databaseHandler
from icecream                                   import ic

# -----------------------------------------------------------------------------------------------------------------------
def create_dataset(p, PDATA, patients):
    for patient in patients:
        ic(patient)
        pimage = os.path.join(PDATA, 'images', patient)
        dataHandler = databaseHandler(p, pimage)
        dataHandler()

# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- using a parser with set_parameters.py allows us to package_core several processes with different set_parameters.py on the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of parameters required to execute the code.')

    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])

    # --- get parameters
    p = param.setParameters()

    # --- launch process
    patients = os.listdir(os.path.join(p.PDATA, "images"))
    patients.sort()

    create_dataset(p, p.PDATA, patients)

# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
