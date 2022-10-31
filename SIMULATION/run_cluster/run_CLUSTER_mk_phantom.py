import os
import subprocess
from icecream import ic


# -----------------------------------------------------------------------------------------------------------------------
def make_dir(path):

    try:
        os.makedirs(path)
    except OSError as error:
        print(error)

# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    path_shell = '/home/laine/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/run_cluster/shell/cluster_make_phantom.sh'
    pdata      = '/home/laine/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/images'
    # pdata      = '/home/laine/PROJECTS_IO/DATA/MCMASTER/CROPPED/CAMO_study_cropped/images'
    pres       = '/home/laine/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER'
    #pres       = '/home/laine/PROJECTS_IO/SIMULATION/MCMASTER/CAMO_STUDY_V2'
    pjson      = '/home/laine/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/parameters/set_parameters_template.json'
   
    fname       = os.listdir(pdata)
    fname.sort()

    # --- PARAMETERS
    info = '_3D'                        # added information to file name
    
    for name in fname[:5]:
        pres_ = os.path.join(pres, name.split('.')[0])
        make_dir(pres_)
        ic(name)
        subprocess.run(['sh',path_shell,pdata,name,pres,pjson,info])
