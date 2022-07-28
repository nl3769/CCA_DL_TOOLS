import numpy as np

def add_seg(LI, MA, I_org, I_gan):
    
    nb_img = len(LI) # add condition to stop if list have different size
    I_org_out = []
    I_gan_out = []
    
    for id in range(nb_img):
        LI_ = LI[id]
        MA_ = MA[id]
        id_LI = np.nonzero(LI_)
        id_MA = np.nonzero(MA_)

        for idi in np.nditer(id_LI):
            I_org[id][int(LI_[idi]), idi] = 255
            I_gan[id][int(LI_[idi]), idi] = 255
        
        for idi in np.nditer(id_MA):
            I_org[id][int(MA_[idi]), idi] = 255
            I_gan[id][int(MA_[idi]), idi] = 255
