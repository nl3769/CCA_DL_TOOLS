'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import numpy                                as np
import matplotlib.pyplot                    as plt

from scipy                                  import ndimage
from skimage.measure                        import label
from numba                                  import jit

# ----------------------------------------------------------------------------------------------------------------------
class IMCExtractor():

    def __init__(self, adventitia_part):

        self.in_masks           = np.empty(())
        self.out_masks          = np.empty(())
        self.batch_size         = 0
        self.mask_size          = 0
        self.seed               = []
        self.adventitia_part    = adventitia_part
        self.zCF                = 0
        self.IMC                = 0
        self.seg                = 0

    # ------------------------------------------------------------------------------------------------------------------
    def update(self, masks, zCF):

        # self.in_masks = np.array(masks.detach().cpu())
        self.in_masks   = masks.detach().cpu()
        self.out_masks  = np.zeros(self.in_masks.shape)
        self.batch_size = self.in_masks.shape[0]
        self.mask_size  = self.in_masks.shape[2:]
        self.seed       = []
        self.zCF        = zCF
        self.IMC        = np.zeros(self.batch_size)
        self.seg        = np.zeros((self.batch_size,) + (self.mask_size[1],) + (2,))

    # ------------------------------------------------------------------------------------------------------------------
    def get_biggest_connected_region(self):
        """ Get the biggest connected region of the input batch. """

        for id_batch in range(self.batch_size):

            img_ = self.in_masks[id_batch, ]
            img_ = img_.squeeze()
            img_fill_holes_ = ndimage.binary_fill_holes(img_).astype(int)
            label_image, nbLabels = label(img_fill_holes_, return_num=True)
            regionSize = []
            if nbLabels > 1:
                for k in range(1, nbLabels + 1):
                    regionSize.append(np.sum(label_image == k))
                regionSize = np.asarray(regionSize)
                idx = np.argmax(regionSize) + 1
                label_image[label_image != idx] = 0
                label_image[label_image == idx] = 1
                img_fill_holes_ = label_image
            elif nbLabels < 1:
                img_fill_holes_ = np.zeros(img_.shape)
                img_fill_holes_[5:-5, :] = 1

            self.out_masks[id_batch, 0, ] = img_fill_holes_


    # ------------------------------------------------------------------------------------------------------------------
    def get_seeds(self):
        """ Compute center of mass of the mask and return the x and y coordinates. """

        for id_batch in range(self.batch_size):
            white_pixels = np.array(np.where(self.out_masks[id_batch, 0, ] == 1))
            seed_ = (round(np.mean(white_pixels[0, ])), round(np.mean(white_pixels[1, ])))
            self.seed.append(seed_)

    # ------------------------------------------------------------------------------------------------------------------
    def IMC_extraction(self):
        """ Compute the  position of the LI interface, MA interface and compute the intima media thickness. """

        for id_batch in range(self.batch_size):
            limit = self.mask_size[0]
            xl, xr = 0, self.mask_size[1]
            neighbors = 30
            extract_LI_CL(self.seed[0], xl, self.out_masks[id_batch, 0, ], neighbors, limit, self.seg, id_batch)
            extract_LI_CR(self.seed[0], xr, self.out_masks[id_batch, 0, ], neighbors, limit, self.seg, id_batch)
            extract_MA_CL(self.seed[0], xl, self.out_masks[id_batch, 0, ], neighbors, limit, self.seg, id_batch)
            extract_MA_CR(self.seed[0], xr, self.out_masks[id_batch, 0, ], neighbors, limit, self.seg, id_batch)
            self.IMC[id_batch] = np.mean(np.abs(self.seg[id_batch, :, 1] - self.seg[id_batch, :, 0])) * self.zCF[id_batch]

    # ------------------------------------------------------------------------------------------------------------------
    def generate_masks(self):

        masks = np.zeros(self.in_masks.shape)

        for id_batch in range(self.batch_size):
            pos_LI = self.seg[id_batch, :, 0]
            pos_MA = self.seg[id_batch, :, 1] + np.round(np.array(self.adventitia_part * 1e-3 / self.zCF[id_batch]))
            for i in range(self.mask_size[1]):
                if pos_MA[i] < self.mask_size[0]:
                    masks[id_batch, 0, int(pos_LI[i]):int(pos_MA[i]), i] = 1
                else:
                    masks[id_batch, 0, int(pos_LI[i]):(self.mask_size[0]-1), i] = 1

        return masks

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """ Compute the biggest connected region, then compute intima media complexe interfaces. """

        self.get_biggest_connected_region()
        self.get_seeds()
        self.IMC_extraction()
        masks = self.generate_masks()
        return masks, self.out_masks, self.IMC


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- EXTRACT INTERFACES ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


@jit(nopython=True)
def extract_MA_CL(seed, x_l, mask, neighbors, limit, seg, id_batch):
    """ Computes the MA interface from the center to the left. """

    j = 0
    height = limit
    for i in range(seed[1], x_l - 1, -1):
        condition = True
        while condition == True:
            if (j < limit and mask[height - 1 - j, i] == 1):
                seg[id_batch, i, 1] = height - 1 - j
                condition = False
            elif j == limit:
                seg[id_batch, i, 1] = seg[id_batch, i + 1, 1]
                condition = False
            j += 1
        j -= neighbors
        limit = j + 2 * neighbors

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def extract_MA_CR(seed, x_r, mask, neighbors, limit, seg, id_batch):
    """ Computes the MA interface from the center to the right. """

    j = 0
    height = limit
    for i in range(seed[1], x_r, 1):
        condition = True
        while condition == True:
            if (j < limit and mask[height - 1 - j, i] == 1):
                seg[id_batch, i, 1] = height - 1 - j
                condition = False
            elif j == limit:
                seg[id_batch, i, 1] = seg[id_batch, i - 1, 1]
                condition = False
            j += 1
        j -= neighbors
        limit = j + 2 * neighbors

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def extract_LI_CR(seed, x_r, mask, neighbors, limit, seg, id_batch):
    """ Computes the LI interface from the center to the left. """

    j = 0
    limit_ = limit
    for i in range(seed[1], x_r, 1):
        condition = True
        while condition == True:
            if (j < limit_ and mask[j, i] == 1):
                seg[id_batch, i, 0] = j
                condition = False
            elif j == limit_:
                seg[id_batch, i, 0] = seg[id_batch, i - 1, 0]
                condition = False
            j += 1
        j -= neighbors
        limit_ = j + 2 * neighbors

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def extract_LI_CL(seed, x_l, mask, neighbors, limit, seg, id_batch):
    """ Computes the LI interface from the center to the left. """

    j = 0
    limit_ = limit
    for i in range(seed[1], x_l - 1, -1):
        condition = True
        while condition == True:
            if (j < limit and mask[j, i] == 1):
                seg[id_batch, i, 0] = j
                condition = False
            elif j == limit_:
                seg[id_batch, i, 0] = seg[id_batch, i + 1, 0]
                condition = False
            j += 1
        j -= neighbors
        limit_ = j + 2 * neighbors

# ----------------------------------------------------------------------------------------------------------------------