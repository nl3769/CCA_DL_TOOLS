import os

import torch.nn                         as nn

from package_utils.utils                import check_dir

class Parameters:

    def __init__(
        self,
        MODEL_NAME,
        PDATA,
        PMODEL,
        DATABASE,
        NB_SAVE,
        IMAGE_NORMALIZATION,
        IMG_SIZE,
        PATH_RES,
        KERNEL_SIZE,
        PADDING,
        USE_BIAS,
        UPCONV,
        NGF,
        NB_LAYERS,
        OUTPUT_ACTIVATION
        ):

        self.MODEL_NAME = MODEL_NAME
        self.PDATA = PDATA
        self.PATH_SAVE_MODEL = PMODEL
        self.DATABASE = DATABASE
        self.NB_SAVE = NB_SAVE
        self.IMAGE_NORMALIZATION = IMAGE_NORMALIZATION
        self.IMG_SIZE = IMG_SIZE
        self.PATH_RES = PATH_RES
        self.KERNEL_SIZE = KERNEL_SIZE
        self.PADDING = PADDING
        self.USE_BIAS = USE_BIAS
        self.UPCONV = UPCONV
        self.NGF = NGF
        self.NB_LAYERS = NB_LAYERS
        self.OUTPUT_ACTIVATION = OUTPUT_ACTIVATION
        self.RESTORE_CHECKPOINT = True


        self.PATH_RES_ORG = os.path.join(PATH_RES, 'img_org')
        self.PATH_RES_GAN = os.path.join(PATH_RES, 'img_gan')
        self.PATH_RES_SIM = os.path.join(PATH_RES, 'img_sim')

        # --- create directories
        check_dir(self.PATH_RES_ORG)
        check_dir(self.PATH_RES_GAN)
        check_dir(self.PATH_RES_SIM)