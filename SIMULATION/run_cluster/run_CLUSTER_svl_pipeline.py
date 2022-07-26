import os
import subprocess
import glob
import json
import scipy.io
from icecream import ic


# ----------------------------------------------------------------------------------------------------------------------
def find_string(files: list, string: str):

    sub_string=string.split('_')[-1];

    for k in range(len(files)):
        if sub_string in files[k]:
            param_id=files[k]
            break
    return param_id

# ----------------------------------------------------------------------------------------------------------------------
def get_mat_files(files):
    list_=[]
    for file in files:
        if '.mat' in file:
            list_.append(file)
    return list_

# ----------------------------------------------------------------------------------------------------------------------
def get_param(ppath):

    file = glob.glob(os.path.join(ppath, '*.json'))
    with open(os.path.join(ppath, file[0]), 'r') as f:
        PARAM = json.load(f)
    info_list = ['phantom_name', 'Nelements', 'Nactive', 'mode', 'nb_tx']
    info = {}

    for key in info_list[:-1]:
            info[key] = PARAM[key]

    if info['mode'][0] == 1:
        info['nb_tx'] = info['Nelements'] - info['Nactive']
    elif info['mode'][1] == 1:
        info['nb_tx'] = info['Nelements']

    pname = os.path.join(ppath, file[0])

    return info['phantom_name'], info['nb_tx'], pname

# ----------------------------------------------------------------------------------------------------------------------
def get_phantom_name(phpath, sub_str):

    file = glob.glob(os.path.join(phpath, '*.mat'))
    file = [i for i in file if sub_str in i]
    return file[0]

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    """ Execute le code sur le cluster pour un plusieurs fantomes contenu dans un même répertoire. """

    path_shell    = '/home/laine/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/run_cluster/shell/cluster_full_pipeline.sh'
    pname         = '/home/laine/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER'
    # path_shell = '/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/run_cluster/shell/cluster_full_pipeline.sh'
    # pname = '/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER'
    patients = os.listdir(pname)
    patients.sort()
    
    for patient in patients[10:15]:
        fname = os.path.join(pname, patient)
        folders = os.listdir(fname)
        folders.sort()
        ic(fname)
        
        for exp in folders[:10]:
            ic(exp)
           
            path_param      = os.path.join(fname, exp, 'parameters')
            path_phantom    = os.path.join(fname, exp, 'phantom')
            pRF             = os.path.join(fname, exp, 'raw_data')
            
            ph_sub_str, nb_tx, pname_ = get_param(path_param)
            phname                    = get_phantom_name(path_phantom, ph_sub_str)
            log_name                  = ph_sub_str
            pres                      = os.path.join(fname, exp)

            # --- run
            subprocess.run(['sh', path_shell, pname_, phname, 'true', log_name, str(nb_tx), pRF, pres])
