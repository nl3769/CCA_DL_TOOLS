'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import argparse
import importlib
import os
from tqdm                                   import tqdm
from package_database.databasePreparation   import databasePreparation
from icecream                               import ic

# -----------------------------------------------------------------------------------------------------------------------
def prepare_dataset(p, simulation):
    pdata = p.PDATA
    for simu in simulation:
        ic(simu)
        p.PDATA = os.path.join(pdata, simu)
        dataPreparation = databasePreparation(p)
        dataPreparation()

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
    simulation = os.listdir(p.PDATA)
    simulation.sort()
    prepare_dataset(p, simulation)

# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
