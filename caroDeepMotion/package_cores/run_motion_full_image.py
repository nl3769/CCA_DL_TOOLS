import os
import torch
import argparse
import importlib
import package_network.utils                as pnu
from package_motion.motionFullImageHandler  import motionFullImgHandler

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- using a parser with set_parameters.py allows us to package_core several processes with different set_parameters.py on the cluster
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netEncoder, netFlow = pnu.load_model_flow(p)
    netEncoder = netEncoder.to(device)
    netFlow = netFlow.to(device)
    netFlow.eval()
    netEncoder.eval()

    simulation = os.listdir(p.PDATA)
    if 'backup_parameters' in simulation:
        simulation.remove('backup_parameters')
    simulation.sort()
    pdata = p.PDATA
    psave = p.PSAVE
    for simu in simulation:
        p.PDATA = os.path.join(pdata, simu)
        p.PSAVE = os.path.join(psave, simu)
        motion = motionFullImgHandler(p, netEncoder, netFlow, device)
        motion()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
