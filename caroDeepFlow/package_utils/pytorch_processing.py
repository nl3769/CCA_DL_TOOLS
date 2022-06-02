'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch

# ----------------------------------------------------------------------------------------------------------------------
def treshold_mask(I, tresh):
    """ Binarize predicted . """

    I[I > tresh] = 1
    I[I < 1] = 0

    return I

# ----------------------------------------------------------------------------------------------------------------------