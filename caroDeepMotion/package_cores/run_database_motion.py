'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import argparse
import importlib
import os
import numpy                                        as np
from package_database.databaseMotionHandler         import databaseHandler
from multiprocessing                                import Process

# -----------------------------------------------------------------------------------------------------------------------
def call(dataHandler):
    dataHandler()

# -----------------------------------------------------------------------------------------------------------------------
def create_dataset(p, simulation):

    pdata = p.PDATA
    # --- split data for multiprocessing
    nb_process = 8
    id_step = 0
    nb_step = int(np.floor(len(simulation) / nb_process))
    for id in range(nb_step):
        process = []
        for id_rel, simu in enumerate(simulation[id_step*nb_process:id_step*nb_process+nb_process]):
            print(str(id * nb_process + id_rel) + ' | ' + simu )
            p.PDATA = os.path.join(pdata, simu)
            dataHandler = databaseHandler(p)
            pc = Process(target=call, args=(dataHandler,))
            process.append(pc)
            pc.start()
        for i in process:
            i.join()
        id_step += 1
    process = []
    # --- we process the last data (because nb_process is probably not a multiple of the number of images)
    for id_rel, simu in enumerate(simulation[id_step * nb_process:]):
        print(str(nb_step * nb_process + id_rel) + ' | ' + simu)
        p.PDATA = os.path.join(pdata, simu)
        dataHandler = databaseHandler(p)
        pc = Process(target=call, args=(dataHandler,))
        process.append(pc)
        pc.start()

# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- using a parser with set_parameters.py allows us to package_core several processes with different set_parameters.py on the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    # --- get parameters
    p = param.setParameters()
    # --- launch process
    simulation = os.listdir(p.PDATA)
    if 'backup_parameters' in simulation:
        simulation.remove('backup_parameters')
    simulation.sort()
    create_dataset(p, simulation)

# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This function generates the database for motion: it splits data into isotropic patches with a specific overlap.
    """
    main()
