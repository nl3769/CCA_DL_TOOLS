import os
import subprocess
import glob
import json
import scipy.io
from icecream import ic

# ----------------------------------------------------------------------------------------------------------------------
def get_idx_tx_event(pRF, nb_event):

    tx_events = os.listdir(pRF)
    tx_events = [int(key.split('_')[-1].split('.')[0]) for key in tx_events]

    if nb_event == len(tx_events):
        non_tx_events = None
    else:
        non_tx_events = []
        for id in range(1, nb_event+1):
            if id not in tx_events:
                non_tx_events.append(id)

    return non_tx_events
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

    path_shell    = '/home/laine/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/run_cluster/shell/cluster_failures.sh'
    pname         = '/home/laine/cluster/PROJECTS_IO/SIMULATION/RU_2022_10_21/'
    nb_tx         = 192

    patients = os.listdir(pname)
    patients.sort()

    for patient in patients[2:]:
        fname = os.path.join(pname, patient)
        folders = os.listdir(fname)
        folders.sort()
        ic(fname)
        
        for exp in folders:
            ic(exp)
           
            path_param      = os.path.join(fname, exp, 'parameters')
            path_phantom    = os.path.join(fname, exp, 'phantom')
            pRF             = os.path.join(fname, exp, 'raw_data')
            
            ph_sub_str, nb_tx, fparam = get_param(path_param)
            phname                    = get_phantom_name(path_phantom, ph_sub_str)
            pres                      = os.path.join(fname, exp)

            id_tx = get_idx_tx_event(os.path.join(fname, exp, 'raw_data', 'raw_'), 128)
            a=1
            # --- run
            for id in id_tx:
                log_name = "tx_" + str(id) + "_" + \
                           ph_sub_str.split('_')[1] + "_" + ph_sub_str.split('_')[2] + \
                           "_" + ph_sub_str.split('_')[4] + "_" + ph_sub_str.split('_')[5]
                subprocess.run(['sh', path_shell, fparam, phname, log_name, str(id)])
