import os
import argparse
import importlib
import package_utils.load_data                  as puld
import package_utils.writter                    as puw
from package_utils.convert_seg_to_numpy         import convert_seg_to_numpy
from package_processing.compute_IMC             import compute_IMC
# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- get segmentation path
    pseg = os.listdir(p.PSEG)
    pLI = [key for key in pseg if 'LI' in key]
    pMA = [key for key in pseg if 'MA' in key]
    pLI.sort()
    pMA.sort()
    
    # --- get segmentation values
    LI = puld.get_seg(p.PSEG, pLI)
    MA = puld.get_seg(p.PSEG, pMA)
    
    # --- get CF
    with open(p.PCF, 'r') as f:
        CF = f.readlines()
    CF = float(CF[0])

    # --- convert to numpy
    LI, MA = convert_seg_to_numpy(LI, MA, p.ROI, p.PBORDERS)
    puw.save_np_to_pickle(LI*CF, os.path.join(p.PPICKLE, 'LI_meters.pkl'))
    puw.save_np_to_pickle(MA*CF, os.path.join(p.PPICKLE, 'MA_meters.pkl'))

    compute_IMC(LI*CF, MA*CF, p.PFIGURE)
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
