import argparse
import importlib
import cv2
import torch

import package_network.utils                    as pnu
import numpy                                    as np
import matplotlib.pyplot                        as plt

from PIL                                        import Image

# ----------------------------------------------------------------------------------------------------------------------
def load_image(path):

    I = np.mean(np.array(Image.open(path)), axis=-1)
    I = cv2.resize(I, (128, 256), interpolation=cv2.INTER_LINEAR)

    I = I[None, None, ]
    return I

# ----------------------------------------------------------------------------------------------------------------------
def load_OF(path):
    """ Read .flo file in Middlebury format"""

    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            OF = np.resize(data, (int(h), int(w), 2))
            x_coef = 128 / OF.shape[0]
            z_coef = 256 / OF.shape[1]
            OF = cv2.resize(OF, (128, 256), interpolation=cv2.INTER_LINEAR)

            OF[..., 0] = OF[..., 0] * x_coef
            OF[..., 1] = OF[..., 1] * z_coef
            OF = np.moveaxis(OF, -1, 0)

            return OF

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- get model
    netEncoder, netFlow = pnu.load_model_flow(p)

    # --- get image
    pI1 = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/FlyingChairs/data/00001_img1.ppm"
    pI2 = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/FlyingChairs/data/00001_img2.ppm"
    pMotion = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/FlyingChairs/data/00001_flow.flo"

    I1 = torch.tensor(load_image(pI1)).float()
    I2 = torch.tensor(load_image(pI2)).float()

    motion = load_OF(pMotion)
    fmap1, skc1, fmap2, skc2 = netEncoder(I1, I2)
    mask = torch.ones(I1.shape)
    flow_pred = netFlow(I1, fmap1, fmap2, mask)
    motion_pred = flow_pred[-1].detach().numpy()

    pred = np.sqrt(np.power(motion_pred[0, 0, ], 2) + np.power(motion_pred[0, 1, ], 2))
    GT = np.sqrt(np.power(motion[0, ], 2) + np.power(motion[1, ], 2))

    plt.figure()
    plt.subplot2grid((1, 2), (0, 0), colspan=1)
    plt.imshow(pred)
    plt.title('pred')
    plt.colorbar()

    plt.subplot2grid((1, 2), (0, 1), colspan=1)
    plt.imshow(GT)
    plt.title('GT')
    plt.colorbar()

    plt.show()
    a=1
# ----------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()