import os
import argparse
import importlib

from package_utils.load_data                import load_pickle
from package_metrics.compute_metrics        import compute_MAE

# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()
    
    # --- get resuls
    LI = {}
    MA = {}
    IMT = {}
    for key in p.PDATA.keys():
        LI[key]  = load_pickle(os.path.join(p.PDATA[key], 'val_seg', 'LI_meters.pkl'))
        MA[key]  = load_pickle(os.path.join(p.PDATA[key], 'val_seg', 'MA_meters.pkl'))
        IMT[key] = MA[key]- LI[key]
    
    # --- compute errors
    keys = list(LI.keys())
    nb_keys = len(keys)
    for i, key in enumerate(keys):
        if i >= 0 and i < (nb_keys-1):
            for idx in range(i+1, nb_keys):
                LI_mean, LI_std     = compute_MAE(LI[keys[i]], LI[keys[idx]]) 
                MA_mean, MA_std     = compute_MAE(MA[keys[i]], MA[keys[idx]]) 
                IMT_mean, IMT_std   = compute_MAE(IMT[keys[i]], IMT[keys[idx]]) 
                
                with open(os.path.join(p.PRES, keys[i] + '_vs_' + keys[idx] + ".txt"), 'w') as f:
                    f.write("LI mean = " + str(LI_mean*1e6) + '\n' + "LI std = " + str(LI_std*1e6) + '\n')
                    f.write("MA mean = " + str(MA_mean*1e6) + '\n' + "MA std = " + str(MA_std*1e6) + '\n')
                    f.write("IMT mean = " + str(IMT_mean*1e6) + '\n' + "IMT std = " + str(IMT_std*1e6) + '\n')
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
