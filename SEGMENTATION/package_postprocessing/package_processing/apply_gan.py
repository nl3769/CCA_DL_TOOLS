import cv2
import torch

import numpy                                    as np
from package_processing.histogram_extension     import histogram_extension

# -----------------------------------------------------------------------------------------------------------------------
def apply_gan(model, dim, I, interval):

    in_dim = I.shape
    I = cv2.resize(I, dim, interpolation=cv2.INTER_LINEAR)
    I = histogram_extension(I, interval)
    I = I[None, None, ...]
    I = torch.tensor(I)
    I = model(I.float())
    I = I.detach().numpy()    
    I = I.squeeze() 
    I = cv2.resize(I, (in_dim[1], in_dim[0]), interpolation=cv2.INTER_LINEAR)
    
    return I
