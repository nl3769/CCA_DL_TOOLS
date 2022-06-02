import os
import subprocess
from icecream import ic

if __name__ == '__main__':

    path_shell = '/sps/creatis/nlaine/REPO/carotid_US_DL_tool/SIMULATION/shell_scripts/IN2P3_run_make_phantom.sh'
    pdata = '/sps/creatis/nlaine/PROJECTS_IO/DATA/Sequences/HEALTHY_ANDRE_57'
    fname = os.listdir(pdata)
    pres = '/sps/creatis/nlaine/PROJECTS_IO/RESULTS/HEALTHY_ANDRE_57'
    
    # --- PARAMETERS
    software='SIMUS'                  # chose FIELD or SIMUS
    acq_mode='synthetic_aperture'     # chose synthetic aperture or scanline_based
    info='_3D'                        # added information to file name
    nb_img='3'                        # number of image in a sequences
    

    for name in fname[0:10]:
        ic(name)
        ic(path_shell)
        subprocess.run(['sh',path_shell,pdata,name,pres,info,software,acq_mode,nb_img])
