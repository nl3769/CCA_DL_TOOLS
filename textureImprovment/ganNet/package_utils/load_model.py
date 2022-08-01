from package_network.discriminator                  import Discriminator
from package_network.SRGAN                          import GeneratorSR, DiscriminatorSR
from package_network.network_unet                   import Unet
import torch.nn as nn
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
            output_activation   = p.OUTPUT_ACTIVATION
        )

        return discriminator, generator

    if p.MODEL_NAME == 'SRGAN':
        discriminator = DiscriminatorSR(p)
        generator = GeneratorSR(p)

        return discriminator, generator
