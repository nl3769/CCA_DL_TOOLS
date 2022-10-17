import package_utils.fold_handler as fh
import os

class Parameters:

    def __init__(
        self,
        PDATA,
        PRES,
        PMODEL,
        DIM_IMG_GAN,
        INTERVAL,
        NB_LAYERS,
        NGF,
        KERNEL_SIZE,
        PADDING,
        USE_BIAS
):

        self.PDATA          = PDATA
        self.PMODEL         = PMODEL
        self.DIM_IMG_GAN    = DIM_IMG_GAN
        self.INTERVAL       = INTERVAL
        self.NB_LAYERS      = NB_LAYERS
        self.NGF            = NGF
        self.KERNEL_SIZE    = KERNEL_SIZE
        self.PADDING        = PADDING
        self.USE_BIAS       = USE_BIAS
        self.PRES           = os.path.join(PRES, PDATA.split('/')[-1])
        self.PGAN_OUTPUT    = os.path.join(self.PRES, 'gan_output')
        self.PGIF           = os.path.join(self.PRES, 'gif_output')
        self.PORG           = os.path.join(self.PRES, 'org')
        self.PSEQ           = os.path.join(self.PRES, 'seq')
        self.PSEG_GT        = os.path.join(self.PRES, 'seg_gt')


        # --- create directories
        fh.create_dir(self.PGAN_OUTPUT)
        fh.create_dir(self.PGIF)
        fh.create_dir(self.PORG)
        fh.create_dir(self.PSEQ)
        fh.create_dir(self.PSEG_GT)
