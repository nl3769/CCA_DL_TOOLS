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
            output_activation   = p.OUTPUT_ACTIVATION
            )

        return discriminator, generator

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
            output_activation   = p.OUTPUT_ACTIVATION
            )

        return discriminator, generator
    
    if p.MODEL_NAME == 'SRGan':
        discriminator = Discriminator()
        generator = SRGan(
            input_nc    = 1,
            output_nc   = 1,
            kernel_size = p.KERNEL_SIZE,
            padding     = p.PADDING,
            use_bias    = p.USE_BIAS,
            )

        return discriminator, generator
