from package_network.discriminator import Discriminator
from package_network.GAN_UPCONV import GeneratorUNetUPCONV
from package_network.GAN_UPSAMPLE import GeneratorUNetUPSAMPLE
from package_network.GAN_UPSAMPLE_02 import GAN_UPSAMPLE_02
from package_network.SRGAN import GeneratorSR, DiscriminatorSR

def load_model(p):

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
