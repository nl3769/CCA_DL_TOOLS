'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import argparse
import importlib
import os
from package_database.databaseHandler       import databaseHandler
from icecream                               import ic

# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- using a parser with set_parameters.py allows us to run several processes with different set_parameters.py on the cluster
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
    PDATAo = p.PDATA
    for simu in simulation:
        ic(simu)
        p.PDATA = os.path.join(PDATAo, simu)
        dataHandler = databaseHandler(p)
        dataHandler()


# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
