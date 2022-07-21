from package_network.discriminator                  import Discriminator
from package_network.GAN_UPCONV                     import GeneratorUNetUPCONV
from package_network.GAN_UPSAMPLE                   import GeneratorUNetUPSAMPLE
from package_network.GAN_UPSAMPLE_02                import GAN_UPSAMPLE_02
from package_network.SRGAN                          import GeneratorSR, DiscriminatorSR
from package_network.GAN_AI4HEALTH                  import GeneratorUNet_AI4HEALTH
from package_network.network_unet                   import Unet

import torch.nn as nn

def load_model(p):


    if p.MODEL_NAME == 'UnetGZ':
        discriminator = Discriminator()
        generator = Unet(
            input_nc        = 1,
            output_nc       = 1,
            n_layers        = p.NB_LAYERS, 
            ngf             = p.NGF,
            norm_layer      = nn.BatchNorm2d,
            kernel_size     = p.KERNEL_SIZE,
            padding         = p.PADDING,
            activation      = nn.LeakyReLU(0.2, True),
            use_bias        = p.USE_BIAS
        )

        return discriminator, generator

    if p.MODEL_NAME == 'GAN_UPCONV':

        discriminator = Discriminator()
        generator_UNet = GeneratorUNetUPCONV(p)

        return discriminator, generator_UNet

    if p.MODEL_NAME == 'GAN_UPSAMPLE':

        discriminator = Discriminator()
        generator_UNet = GeneratorUNetUPSAMPLE(p)

        return discriminator, generator_UNet
    if p.MODEL_NAME == 'GAN_UPSAMPLE_02':

        discriminator = Discriminator()
        generator_UNet = GAN_UPSAMPLE_02(p)

        return discriminator, generator_UNet

    if p.MODEL_NAME == 'SRGAN':
        discriminator = DiscriminatorSR(p)
        generator = GeneratorSR(p)

        return discriminator, generator

    if p.MODEL_NAME == 'AI4HEALTH':
        discriminator = Discriminator()
        generator = GeneratorUNet_AI4HEALTH()

        return discriminator, generator
