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

    path_shell = '/home/laine/REPOSITORIES/carotid_US_DL_tool/SIMULATION/run_cluster/shell/CLUSTER_run_make_phantom.sh'
    pdata = '/home/laine/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/images'
    fname = os.listdir(pdata)
    fname.sort()
    pres = '/home/laine/PROJECTS_IO/SIMULATION/CUBS'

    # --- PARAMETERS
    software='FIELD'                  # choose FIELD or SIMUS
    acq_mode='synthetic_aperture'     # choose synthetic aperture or scanline_based
    info='_3D'                        # added information to file name
    nb_img='10'                      # number of image in a sequenices
    ic(fname)
    for name in fname[:10]:
        pres_ = os.path.join(pres, name.split('.')[0])
        make_dir(pres_)
        ic(name)
        ic(path_shell)
        subprocess.run(['sh',path_shell,pdata,name,pres_,info,software,acq_mode,nb_img])
