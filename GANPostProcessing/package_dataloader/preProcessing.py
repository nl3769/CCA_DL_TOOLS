import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa
import random

class preProcessing():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p, data_aug = False):

        self.data_aug = data_aug
        self.im_size = p.IMG_SIZE
        self.augmenter = iaa.Sequential([iaa.Fliplr(0.5),
                                         iaa.Flipud(0.5),
                                         iaa.Affine(shear=(-20, 20),
                                                    rotate=(-10, 10),
                                                    translate_px={"y": (-30, 30), "x": (-30, 30)})])
        self.normalization = p.IMAGE_NORMALIZATION

    # ------------------------------------------------------------------------------------------------------------------
    def augmentation(self, org, sim):

        org = np.expand_dims(org, axis=(0, -1))
        sim = np.expand_dims(sim, axis=(0, -1))
        org, sim = self.augmenter(images=org, segmentation_maps=sim)

        org = org.squeeze()
        sim = sim.squeeze()

        return org, sim

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def reshape(org: np.ndarray, sim: np.ndarray, out_dim: tuple):
        """ Reshape image to the desired resolution. it is applied for all images in the database. """

        # --- reshape images
        org = cv2.resize(org, out_dim, interpolation=cv2.INTER_LINEAR)
        sim = cv2.resize(sim, out_dim, interpolation=cv2.INTER_LINEAR)

        return org, sim

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def adapt_dim(org: np.ndarray, sim: np.ndarray):

        """ Adapt dimension to fit pytorch format. """

        sim = np.expand_dims(sim, axis=0)
        org = np.expand_dims(org, axis=0)

        sim = torch.from_numpy(sim).float()
        org = torch.from_numpy(org).float()

        return org, sim

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def normalize(org: torch.Tensor, sim: torch.Tensor):
        """ adapt hisotgram between 0 and 255. """


        sim = sim - torch.min(sim)
        sim = sim / torch.max(sim)

        org = org - torch.min(org)
        org = org / torch.max(org)
        # sim = sim / torch.max(sim)
        # org = org / torch.max(org)

        return org*255, sim*255

    # ----------------------------------------------------------------
    @staticmethod
    def histogram_extension(sample_rec, interval):

        sample_rec = np.array(sample_rec)
        delta = interval[1] - interval[0]
        sample_rec  = sample_rec - sample_rec.min()
        sample_rec  = sample_rec * (delta / sample_rec.max())
        sample_rec  = sample_rec + interval[0]

        return sample_rec

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, org: np.ndarray, sim: np.ndarray):

        org, sim = self.reshape(org, sim, self.im_size)
        org = self.histogram_extension(org, self.normalization)
        sim = self.histogram_extension(sim, self.normalization)
        # if self.data_aug:
        #     org, sim = self.augmentation(org, sim)

        org, sim = self.adapt_dim(org, sim)
        # org, sim = self.normalize(org, sim)

        return org, sim

    # ------------------------------------------------------------------------------------------------------------------
