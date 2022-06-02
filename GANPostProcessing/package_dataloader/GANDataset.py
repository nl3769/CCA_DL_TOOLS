import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio as iio
from torch.utils.data import Dataset

from package_dataloader.preProcessing import preProcessing


class GANDataSet(Dataset):

    def __init__(self, p, data_aug):

        self.img_size = p.IMG_SIZE
        self.is_test = False
        self.org_list = []
        self.sim_list = []
        self.extra_info = []
        self.preprocessing = preProcessing(p, data_aug=data_aug)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):

        # --- modulo, to not be out of the array
        index = index % len(self.org_list)

        # --- read images
        org, sim, = self.read_img(index)

        # --- apply preprocessing
        org, sim = self.preprocessing(org, sim)

        # --- get the name of the sequence
        name = self.org_list[index][0].split('/')[-2]

        return org, sim, name

    # ------------------------------------------------------------------------------------------------------------------
    def read_img(self, index):
        """ Read images. """

        org = iio.imread(self.org_list[index][0])
        sim = iio.imread(self.sim_list[index][0])

        return org, sim

    # ------------------------------------------------------------------------------------------------------------------
    def __rmul__(self, v):

        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list

        return self

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.org_list)

    # ------------------------------------------------------------------------------------------------------------------
