import os
import shutil
from glob                           import glob
import numpy                        as np
import pickle                       as pkl
from tqdm                           import tqdm
"""
This script sort flying chairs: it removes pairs of images with a displacement field higher than TRESHOLD pxl (in norm)
"""

# ----------------------------------------------------------------------------------------------------------------------
def create_dir(path):
    """ Create directory. """
    isExist = os.path.exists(path)

    if not isExist:
        # ---  create a new directory because it does not exist
        os.makedirs(path)

# ----------------------------------------------------------------------------------------------------------------------
def mk_copy(pres, name, seq_id, pseq, correct):

    # --- mk dir for results
    files = os.listdir(pseq)
    files.remove('CF.txt')

    pres = os.path.join(pres, name, seq_id)

    # --- copy CF
    create_dir(pres)
    psource = os.path.join(pseq, 'CF.txt')
    ptarget = os.path.join(pres, 'CF.txt')
    shutil.copy(psource, ptarget)

    # --- copy data
    for file in files:
        create_dir(os.path.join(pres, file))
        for correct_file in correct:
            psource = os.path.join(pseq, file, correct_file)
            ptarget = os.path.join(pres, file, correct_file)
            shutil.copy(psource, ptarget)

# ----------------------------------------------------------------------------------------------------------------------
def handler(path, name, pres, max_motion):

    pseq = os.path.join(path, name)
    nseq = os.listdir(pseq)

    for seq in nseq:
        pdata = os.path.join(pseq, seq)
        files = os.listdir(pdata)

        if 'OF' in files:
            to_copy = []
            pmotion = os.path.join(pdata, 'OF')
            fmotion = os.listdir(pmotion)
            for motion in fmotion:
                with open(os.path.join(pmotion, motion), 'rb') as f:
                    data = pkl.load(f)
                if np.max(data) < max_motion:
                    to_copy.append(motion)
                else:
                    print("delete --> " + os.path.join(pmotion, motion))

        if len(to_copy) > 0:
            mk_copy(pres, name, seq, pdata, to_copy)

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pdata = '/run/media/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/database_training_IMAGENET/'
    # pres = '/run/media/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/database_training_IMAGENET_10_PX'
    pres = '/home/laine/Desktop/database_training_IMAGENET_10_PX'
    simulation = os.listdir(pdata)
    max_motion = 10

    for sim in tqdm(simulation):
        handler(pdata, sim, pres, max_motion)

    # ----------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    main()