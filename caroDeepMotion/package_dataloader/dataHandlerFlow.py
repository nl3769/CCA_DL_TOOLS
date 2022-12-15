import os
import sys

from glob                           import glob
from package_dataloader.FlowLoader  import FlowDataloader

import numpy                        as np

# ----------------------------------------------------------------------------------------------------------------------
def check_dim(res, patient):

    dim = {}
    for key in res.keys():
        dim[key] = len(res[key])

    keys = list(dim.keys())
    init = dim[keys[0]]

    for key in keys[1:]:
        if dim[key] != init:
            sys.exit('Error in check_dim in dataHandler: ' + patient)

# ----------------------------------------------------------------------------------------------------------------------
class dataHandlerFlowFlyingChair(FlowDataloader):

    def __init__(self, param, set):
        super(dataHandlerFlowFlyingChair, self).__init__()

        if set != 'validation':
            files = os.listdir(param.PSPLIT)
            mag = str(param.MAGNITUDE_MAGNITUDE)
            img_fname = [key for key in files if 'images' in key and mag in key][0]
            motion_fname = [key for key in files if 'motion' in key and mag in key][0]

            with open(os.path.join(param.PSPLIT, img_fname), 'r') as f:
                img = f.readlines()
            with open(os.path.join(param.PSPLIT, motion_fname), 'r') as f:
                motion = f.readlines()

            images = [os.path.join(param.PDATA, key.replace('\n', '')) for key in img]
            flows = [os.path.join(param.PDATA, key.replace('\n', '')) for key in motion]

            # images = sorted(glob(os.path.join(param.PDATA, '*.ppm')))
            # flows = sorted(glob(os.path.join(param.PDATA, '*.flo')))
            assert (len(images) // 2 == len(flows))

            '''
            # split_list = np.loadtxt(os.path.join(param.PDATA, 'chairs_split.txt'), dtype=np.int32)
            # for i in range(len(flows)):
            #     xid = split_list[i]
            #     if (set == 'training' and xid == 1) or (set == 'validation' and xid == 2):
            #         self.flow_list += [[flows[i]]]
            #         self.image_list += [[images[2*i], images[2*i+1]]]
            '''

            for i in range(len(flows)):
                self.flow_list += [[flows[i]]]
                self.image_list += [[images[2*i], images[2*i+1]]]

            self.flow_list = self.flow_list
            self.image_list = self.image_list

        # self.flow_list = self.flow_list[:8]
        # self.image_list = self.image_list[:8]