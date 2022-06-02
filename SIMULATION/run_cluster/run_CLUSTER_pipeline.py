import os
import subprocess
import glob
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

    file = glob.glob(os.path.join(ppath, '*.mat'))
    PARAM = scipy.io.loadmat(os.path.join(ppath, file[0]))

    info_list = ['phantom_name', 'Nelements', 'Nactive', 'mode', 'nb_tx']
    info = {}

    for key in info_list[:-1]:
        info[key] = PARAM['p'][key][0,0][0]

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

    """ execute le code sur le cluster pour un plusieurs fantomes contenu dans un même répertoire. """

    path_shell = '/home/laine/REPOSITORIES/carotid_US_DL_tool/SIMULATION/run_cluster/shell/CLUSTER_run_shell_full_pipeline.sh'
    fname = '/home/laine/PROJECTS_IO/SIMULATION/CUBS/tech_004'
    folders = os.listdir(fname)
    folders.sort()
    ic(folders)
    for exp in folders:
        ic(exp)
        path_param=os.path.join(fname, exp, 'parameters')
        path_phantom=os.path.join(fname, exp, 'phantom')

        ph_sub_str, nb_tx, pname = get_param(path_param)
        phname = get_phantom_name(path_phantom, ph_sub_str)
        pRF = os.path.join(fname, exp, 'raw_data')
        log_name = ph_sub_str
        pres = os.path.join(fname, exp)

        # --- display
        ic(path_shell)
        ic(pname)
        ic(phname)
        ic(log_name)
        ic(nb_tx[0])
        ic(pRF)
        subprocess.run(['sh', path_shell, pname, phname, 'true', log_name, str(nb_tx[0]), pRF, pres])
