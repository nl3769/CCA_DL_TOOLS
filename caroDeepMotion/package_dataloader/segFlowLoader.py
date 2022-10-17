import numpy                as  np
import package_utils.loader as  pul
from torch.utils.data       import        Dataset

import os

# ----------------------------------------------------------------------------------------------------------------------
class segFlowDataloader(Dataset):

    def __init__(self, param):

        self.is_test = False
        self.image_list = []
        self.mask_list = []
        self.CF_list = []
        self.extra_info = []

        # self.preprocessing = preProcessing(p)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        # --- modulo, to not be out of the array
        index = index % len(self.image_list)

        # --- read data
        I1, I2, M1, M2 = self.read_img(index)
        OF = self.read_OF(index)

        CF = self.get_CF(index)

        # --- get the name of the sequence
        name_ = self.image_list[index][0].split('/')[-4:]
        name = ''
        for id, key in enumerate(name_):
            if id  == 0:
                name = key
            else:
                name = os.path.join(name, key)

        return I1, I2, M1, M2, OF, CF, name

    # ------------------------------------------------------------------------------------------------------------------
    def read_img(self, index):
        """ Read images and flow from files. """

        I1 = pul.load_image(self.image_list[index][0])
        I2 = pul.load_image(self.image_list[index][1])
        M1 = pul.load_image(self.mask_list[index][0])
        M2 = pul.load_image(self.mask_list[index][1])

        I1 = np.array(I1).astype(np.float32)[None, ...] / 255
        I2 = np.array(I2).astype(np.float32)[None, ...] / 255
        M1 = np.array(M1).astype(np.float32)[None, ...] / 255
        M2 = np.array(M2).astype(np.float32)[None, ...] / 255

        return I1, I2, M1, M2

    # ------------------------------------------------------------------------------------------------------------------

    def read_OF(self, index):
        OF = pul.load_nii(self.flow_list[index][0])
        OF = OF.get_fdata()
        OF = np.array(OF).astype(np.float32)
        OF = np.transpose(OF, (2, 0, 1))[0::2, ...]

        return OF

    # ------------------------------------------------------------------------------------------------------------------
    def get_CF(self, index):
        """ Read .txt file to get calibration factor. """

        with open(self.CF_list[index][0], 'r') as f:
            CF = f.read()

        xCF = float(CF.split('\n')[0].split(' ')[-1])
        yCF = float(CF.split('\n')[1].split(' ')[-1])
        CF = {"xCF": xCF,
              "yCF": yCF}

        return CF

    # ------------------------------------------------------------------------------------------------------------------
    def __rmul__(self, v):
        self.flow_list  = v * self.flow_list
        self.image_list = v * self.image_list

        return self

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):

        return len(self.image_list)

    # ------------------------------------------------------------------------------------------------------------------
