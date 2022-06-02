import os
import sys
from package_dataloader.segFlowLoader import segFlowDataloader

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
class dataHandler(segFlowDataloader):


    def __init__(self, param, set):
        super(dataHandler, self).__init__(param)

        fsplit = os.listdir(param.PSPLIT)
        fsplit = [key for key in fsplit if set in key]

        with open(os.path.join(param.PSPLIT, fsplit[0]), 'r') as f:
            path = f.readlines()

        path = [key.split('\n')[0] for key in path]

        # --- subfolders
        subfolds = ['I1', 'I2', 'M1', 'M2', 'OF']

        # --- get path
        for patient in path:
            seq = os.listdir(patient)
            seq.sort()
            for id_seq in seq:
                path_ = {}
                for subfold in subfolds:
                    path_[subfold] = os.listdir(os.path.join(patient, id_seq, subfold))
                    path_[subfold].sort()

                check_dim(path_, patient)
                pCF = os.path.join(patient, id_seq, 'CF.txt')

                for fpath in path_['I1']:
                    pI1 = os.path.join(patient, id_seq, 'I1', fpath)
                    pI2 = os.path.join(patient, id_seq, 'I2', fpath)
                    pM1 = os.path.join(patient, id_seq, 'M1', fpath)
                    pM2 = os.path.join(patient, id_seq, 'M2', fpath)
                    pOF = os.path.join(patient, id_seq, 'OF', fpath.replace('png', 'nii'))

                    self.image_list.append([pI1, pI2])
                    self.mask_list.append([pM1, pM2])
                    self.flow_list.append([pOF])
                    self.CF_list.append([pCF])

        self.image_list = self.image_list
        self.mask_list = self.mask_list
        self.flow_list = self.flow_list