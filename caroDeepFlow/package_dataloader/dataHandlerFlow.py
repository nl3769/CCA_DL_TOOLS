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
        super(dataHandlerFlowFlyingChair, self).__init__(param)

        images = sorted(glob(os.path.join(param.PSPLIT, '*.ppm')))
        flows = sorted(glob(os.path.join(param.PSPLIT, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt(os.path.join(param.PSPLIT, 'chairs_split.txt'), dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (set=='training' and xid==1) or (set=='validation' and xid==2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2*i], images[2*i+1]]]

        # self.flow_list = self.flow_list[:32]
        # self.image_list = self.image_list[:32]

