import os
import nibabel              as nib
import matplotlib.pyplot    as plt
import pickle               as pkl
import numpy                as np


def get_fnames(pdata, files, substr):
    fnames = []

    for file in files[1:]:
        files_ = os.listdir(os.path.join(pdata, file, substr))
        files_ = [fname for fname in files_ if 'OF' in fname]
        fnames.append(os.path.join(pdata, file, substr, files_[0]))

    return fnames

if __name__ == "__main__":

    pdata = "/home/laine/Desktop/pts_tracking_pht/CAMO01_image1/"
    substr = "phantom"
    files = os.listdir(pdata)
    files.sort()

    fnames = get_fnames(pdata, files, substr)

    nimg = nib.load(fnames[0])
    I = nimg.get_data()
    motion_pkl = np.zeros(I.shape + (len(fnames),))
    for id, fname in enumerate(fnames):
        nimg = nib.load(fname)
        I = nimg.get_data()
        motion_pkl[..., id] = I

    pres = '/home/laine/Desktop/motion.pkl'
    with open(pres, 'wb') as f:
        pkl.dump(motion_pkl, f)
