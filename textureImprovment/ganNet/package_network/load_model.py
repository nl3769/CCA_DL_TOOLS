import os
import torch

from package_network.network_discriminator                  import Discriminator
from package_network.network_SRGan                          import SRGan
from package_network.network_unet                           import Unet
from package_network.network_dilatedUnet                    import dilatedUnet

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def load_model(p):


    if p.MODEL_NAME == 'unet':
        discriminator = Discriminator()
        generator = Unet(
            input_nc            = 1,
            output_nc           = 1,
            n_layers            = p.NB_LAYERS,
            ngf                 = p.NGF,
            kernel_size         = p.KERNEL_SIZE,
            padding             = p.PADDING,
            use_bias            = p.USE_BIAS,
            output_activation   = p.OUTPUT_ACTIVATION,
            upconv              = p.UPCONV
            )

    if p.MODEL_NAME == 'dilatedUnet':
        discriminator = Discriminator()
        generator = dilatedUnet(
            input_nc            = 1,
            output_nc           = 1,
            n_layers            = p.NB_LAYERS,
            ngf                 = p.NGF,
            kernel_size         = p.KERNEL_SIZE,
            padding             = p.PADDING,
            use_bias            = p.USE_BIAS,
            output_activation   = p.OUTPUT_ACTIVATION,
            upconv              = p.UPCONV
            )

    if p.MODEL_NAME == 'SRGan':
        discriminator = Discriminator()
        generator = SRGan(
            input_nc    = 1,
            output_nc   = 1,
            kernel_size = p.KERNEL_SIZE,
            padding     = p.PADDING,
            use_bias    = p.USE_BIAS,
            )

    if p.RESTORE_CHECKPOINT:
        nmodel = get_path_model(p.PATH_SAVE_MODEL, 'training')
        if nmodel is not None and len(nmodel) == 1:
            pmodel = os.path.join(p.PATH_SAVE_MODEL, nmodel[0])
            generator.load_state_dict(torch.load(pmodel))

    return discriminator, generator

# -----------------------------------------------------------------------------------------------------------------------
def get_path_model(pres, set):
    """ Get models name. """

    models = os.listdir(pres)
    
    if len(models) == 0:
        models = None
    else:
        models = [key for key in models if set in key]

    return models
