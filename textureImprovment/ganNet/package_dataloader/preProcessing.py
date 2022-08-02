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
        self.interval = p.IMAGE_NORMALIZATION
        self.augmenter = iaa.Sequential([iaa.Fliplr(0.5),
                                         iaa.Flipud(0.5),
                                         # iia.scale{"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                         iaa.Affine(shear=(-5, 5),
                                                    rotate=(-5, 5),
                                                    translate_px={"y": (-30, 30), "x": (-30, 30)})])

    # ------------------------------------------------------------------------------------------------------------------
    def augmentation(self, org, sim):

        org = np.array(org.squeeze())
        sim = np.array(sim.squeeze())
        augseq_det = self.augmenter.to_deterministic()
        org, sim = augseq_det(images=org), augseq_det(images=sim)
        org, sim = np.expand_dims(org, axis=0), np.expand_dims(sim, axis=0)

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
        org = self.histogram_extension(org, self.interval)
        org, sim = self.adapt_dim(org, sim)
        sim = self.histogram_extension(sim, self.interval)
        if self.data_aug:
            org, sim = self.augmentation(org, sim)
        
        org = torch.tensor(org)
        sim = torch.tensor(sim)
        
        return org.clone().detach(), sim.clone().detach()

    # ------------------------------------------------------------------------------------------------------------------
