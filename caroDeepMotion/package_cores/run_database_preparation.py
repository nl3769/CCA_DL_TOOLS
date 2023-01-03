'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import argparse
import importlib
import os
import numpy                                as np
from package_database.databasePreparation   import databasePreparation
from icecream                               import ic
from multiprocessing                        import Process

# -----------------------------------------------------------------------------------------------------------------------
def call(dataPreparation):
    dataPreparation()

# -----------------------------------------------------------------------------------------------------------------------
def prepare_dataset(p, simulation):
    pdata = p.PDATA
    # simulation = simulation[326:]

    # --- split data for multiprocessing
    nb_process = 8
    id_step = 0
    nb_step = int(np.floor(len(simulation) / nb_process))

    for id in range(nb_step):
        process = []
        for id_rel, simu in enumerate(simulation[id_step*nb_process:id_step*nb_process+nb_process]):
            print(str(id * nb_process + id_rel) + ' | ' + simu )
            p.PDATA = os.path.join(pdata, simu)
            dataPreparation = databasePreparation(p)
            pc = Process(target=call, args=(dataPreparation,))
            process.append(pc)
            pc.start()

        for i in process:
            i.join()

        id_step += 1

    process = []
    for id_rel, simu in enumerate(simulation[id_step * nb_process:]):
        print(str(nb_step * nb_process + id_rel) + ' | ' + simu)
        p.PDATA = os.path.join(pdata, simu)
        dataPreparation = databasePreparation(p)
        pc = Process(target=call, args=(dataPreparation,))
        process.append(pc)
        pc.start()

    for i in process:
        i.join()

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
