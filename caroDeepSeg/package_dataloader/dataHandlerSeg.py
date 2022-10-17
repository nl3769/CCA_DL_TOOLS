import os
import sys
from package_dataloader.segLoader import segDataloader

# ----------------------------------------------------------------------------------------------------------------------
def check_dim(res, patient):
    """We check if the result of each folder is the same (because it works by pair

    Args:
        res (dict): dictionnary which contains the name of all images in the subset
        patients (patient): name of the patient in case of error

    Returns:
        Nothing, it shut the program down if dimensions don't match
    """

    dim = {}
    for key in res.keys():
        dim[key] = len(res[key])

    keys = list(dim.keys())
    init = dim[keys[0]]

    for key in keys[1:]:
        if dim[key] != init:
            sys.exit('Error in check_dim in dataHandler: ' + patient)

# ----------------------------------------------------------------------------------------------------------------------
class dataHandlerSegInSilico(segDataloader):
    '''
    Get the patient's name depending on how we save the results of the simulation. See the database for a better understanding.
    As it is a child class of segDataloader, we only need the constructor of the class.
    '''

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, param, set):
        super(dataHandlerSegInSilico, self).__init__(param)

        # --- get name of patient of the desired subset
        fsplit = os.listdir(param.PSPLIT)
        fsplit = [key for key in fsplit if set in key]
        with open(os.path.join(param.PSPLIT, fsplit[0]), 'r') as f:
            path = f.readlines()

        # --- remove unused string
        path = [os.path.join(param.PDATA, key.split('\n')[0]) for key in path]

        # --- subfolders
        subfolds = ['I1', 'I2', 'M1', 'M2']

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

                    self.image_list.append([pI1])
                    self.image_list.append([pI2])
                    self.mask_list.append([pM1])
                    self.mask_list.append([pM2])
                    self.CF_list.append([pCF])
                    self.CF_list.append([pCF])

        #self.image_list = self.image_list[:10]
        #self.mask_list = self.mask_list[:10]
        #self.CF_list = self.CF_list[:10]

# ----------------------------------------------------------------------------------------------------------------------
class dataHandlerSegCubs(segDataloader):
    '''
    Get the patient's name depending on how we data are saved according to CUBS. See the database for a better understanding.
    As it is a child class of segDataloader, we only need the constructor of the class.
    '''

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, param, set):
        super(dataHandlerSegCubs, self).__init__(param)

        # --- get name of patient of the desired subset
        fsplit = os.listdir(param.PSPLIT)
        fsplit = [key for key in fsplit if set in key]
        with open(os.path.join(param.PSPLIT, fsplit[0]), 'r') as f:
            path = f.readlines()

        # --- remove unused string
        path = [os.path.join(param.PDATA, key.split('\n')[0]) for key in path]

        # --- subfolders
        subfolds = ['I', 'M']

        # --- get path
        for patient in path:
            path_ = {}
            for subfold in subfolds:
                path_[subfold] = os.listdir(os.path.join(patient, subfold))
                path_[subfold].sort()

            check_dim(path_, patient)
            pCF = os.path.join(patient, 'CF.txt')

            for fpath in path_['I']:
                pI = os.path.join(patient, 'I', fpath)
                pM = os.path.join(patient, 'M', fpath)

                self.image_list.append([pI])
                self.mask_list.append([pM])
                self.CF_list.append([pCF])

        # self.image_list = self.image_list[:6]
        # self.mask_list = self.mask_list[:6]
        # self.CF_list = self.CF_list[:6]
# ----------------------------------------------------------------------------------------------------------------------
