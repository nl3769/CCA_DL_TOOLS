import torch
import torch.nn                     as nn
from package_network.network_unet   import Unet

def load_model(p):
    pmodel = p.PMODEL
    model = Unet(
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

    model.load_state_dict(torch.load(pmodel, map_location=torch.device('cpu')))

    return model
