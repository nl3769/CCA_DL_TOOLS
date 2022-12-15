import os
import cv2

from glob                           import glob
from PIL                            import Image
import numpy                        as np
import matplotlib.pyplot            as plt
TRESHOLD = 40

"""
This script sort flying chairs: it removes pairs of images with a displacement field higher than TRESHOLD pxl (in norm)
"""

# ----------------------------------------------------------------------------------------------------------------------

def read_img(pI1):
    """ Read images and flow from files. """

    I1 = np.mean(np.array(Image.open(pI1)), axis=-1)
    I1 = cv2.resize(I1, (256, 256), interpolation=cv2.INTER_LINEAR)
    I1 = np.expand_dims(I1, axis=0)

    return I1

# ----------------------------------------------------------------------------------------------------------------------
def read_OF(path):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)

            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            OF = np.resize(data, (int(h), int(w), 2))
            x_coef = 256 / OF.shape[0]
            z_coef = 256 / OF.shape[1]
            OF = cv2.resize(OF, (256, 256), interpolation=cv2.INTER_LINEAR)

            OF[..., 0] = OF[..., 0] * x_coef
            OF[..., 1] = OF[..., 1] * z_coef
            OF = np.moveaxis(OF, -1, 0)

            return OF

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pdata = '/home/laine/Documents/PROJECTS_IO/DATA/OPTICAL_FLOW/FlyingChairs/data'

    data_ = sorted(glob(os.path.join(pdata, '*.ppm')))
    flows = sorted(glob(os.path.join(pdata, '*.flo')))

    correct_flow = []
    correct_img = []

    assert (len(data_) // 2 == len(flows))

    data = []
    for i in range(len(flows)):
        data.append([data_[2*i], data_[2*i + 1]])
    del data_
    inc = 0
    for pof, pi in zip(flows, data):
        # i1 = read_img(pi[0])
        # i2 = read_img(pi[1])
        of = read_OF(pof)

        val = np.linalg.norm(of, axis=0)

        if val.max() <= TRESHOLD:
            correct_flow.append(pof)
            correct_img.append([pi[0], pi[1]])
            inc += 1

    with open(os.path.join("/home/laine/Desktop", "patient_images_" + str(TRESHOLD) + "_px.txt"), 'w') as f:
        for key in correct_img:
            f.write(key[0].split('/')[-1] + '\n')
            f.write(key[1].split('/')[-1] + '\n')

    with open(os.path.join("/home/laine/Desktop", "patient_motion_" + str(TRESHOLD) + "_px.txt"), 'w') as f:
        for key in correct_flow:
            f.write(key.split('/')[-1] + '\n')

    print('number of valide pairs of images: ', inc)

    # ----------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    main()