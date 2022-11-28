import os
import torch
import sys
import numpy                as  np
import package_utils.loader as  pul
import matplotlib.pyplot    as plt
import imgaug.augmenters    as iaa

from torch.utils.data       import        Dataset

# ----------------------------------------------------------------------------------------------------------------------
class segDataloader(Dataset):

    def __init__(self, param):

        self.is_test = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.CF_list = []
        self.extra_info = []

        # self.augmentation = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomAffine(degrees=10, scale=(0.2, 0.2), translate=(0.2, 0.2), shear=20),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.RandomVerticalFlip(0.5)]
        # )
        self.augmentation = iaa.Sequential([iaa.Fliplr(0.5),
                                            iaa.Flipud(0.5),
                                            iaa.Affine(shear=(-2, 2),
                                                       scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                                       rotate=(-5, 5),
                                                       translate_px={"y": (-20, 20), "x": (-5, 5)})])

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        # --- modulo, to not be out of the array
        index = index % len(self.image_list)

        # --- read data
        I1, M1 = self.read_img(index)
        CF = self.get_CF(index)

        # --- get the name of the sequence
        name_ = self.image_list[index][0].split('/')[-4:]
        name = ''
        for id, key in enumerate(name_):
            if id  == 0:
                name = key
            else:
                name = os.path.join(name, key)

        # # test = self.augmentation(I1)
        # aug = self.augmentation.to_deterministic()
        # I1 = aug(images=I1)
        # M1 = aug(images=M1)
        #
        # # -- treshold the mask
        # M1[M1>0.5] = 1
        # M1[M1 <= 0.5] = 0

        I1, M1 = torch.from_numpy(I1).float(), torch.from_numpy(M1).float()

        return I1, M1, CF, name

    # ------------------------------------------------------------------------------------------------------------------
    def read_img(self, index):
        """ Read images and flow from files. """

        disp = False
        if disp == True:
            print('###############################' + '\n')
            print(self.image_list[index][0]  + '\n')
            print(self.mask_list[index][0]  + '\n')
            print('###############################' + '\n')

        I1 = pul.load_pickle(self.image_list[index][0])
        M1 = pul.load_pickle(self.mask_list[index][0])

        I1 = np.array(I1).astype(np.float32)[None, ...] / (np.max(I1) + sys.float_info.epsilon)
        M1 = np.array(M1).astype(np.float32)[None, ...] / (np.max(M1) + sys.float_info.epsilon)

        return I1, M1

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
