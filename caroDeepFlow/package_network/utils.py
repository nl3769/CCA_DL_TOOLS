import os
import torch

from torch                                          import nn
from package_network.network_RAFTFlow               import NetFlow
from package_network.network_encoder                import NetEncoder
from package_network.network_segDecoder             import NetSegDecoder
from package_network.network_dilatedUnet            import dilatedUnet

# ----------------------------------------------------------------------------------------------------------------------
def initialize_weights(m):
    """ Weights intialization. """

    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

# ----------------------------------------------------------------------------------------------------------------------
def get_path_models(pres, keys):
    """ Get models name. """

    models_name = {}
    path = os.path.join(pres, "saved_models")
    if os.path.exists(path):
        models = os.listdir(path)

        if len(models)>0:
            for key in keys:
                for model in models:
                    if model.find(key) != -1 and model.find("val") != -1:
                        models_name[key] = os.path.join(path, model)
        else:
            models_name = None
    else:
        models_name = None

    return models_name

# ----------------------------------------------------------------------------------------------------------------------
def load_model(param):
    """ Load models with associated weights. """

    # ---------------------
    # ---- LOAD MODELS ----
    # ---------------------


    netEncoder  = NetEncoder(param)
    netFlow     = NetFlow(param)
    netSeg      = dilatedUnet(
        input_nc        = 2,
        output_nc       = 2,
        n_layers        = param.NB_LAYERS,
        ngf             = param.NGF,
        kernel_size     = param.KERNEL_SIZE,
        padding         = param.PADDING,
        use_bias        = param.USE_BIAS
        )

    # -----------------------
    # ---- ADAPT WEIGHTS ----
    # -----------------------

    keys = ['netEncoder', 'netSeg', 'netFlow']
    models_name = get_path_models(param.PRES, keys)

    if param.RESTORE_CHECKPOINT:
        if models_name is not None:
            for model_name in models_name.keys():
                if model_name == keys[0]:
                    netEncoder.load_state_dict(torch.load(models_name[model_name]))
                elif model_name == keys[1]:
                    netSeg.load_state_dict(torch.load(models_name[model_name]))
                elif model_name == keys[2]:
                    netFlow.load_state_dict(torch.load(models_name[model_name]))
    else:
        netEncoder.apply(initialize_weights)
        netSeg.apply(initialize_weights)
        netFlow.apply(initialize_weights)

    return netEncoder, netSeg, netFlow

# ----------------------------------------------------------------------------------------------------------------------
def load_model_seg(param):
    """ Load models with associated weights. """

    netSeg = DilatedUnet(in_channels=2, out_channels=2)

    # --- adapt weights
    keys = ['netSeg']
    models_name = get_path_models(param.PRES, keys)

    if param.RESTORE_CHECKPOINT and models_name != None:
        if models_name is not None:
            for model_name in models_name.keys():
                netSeg.load_state_dict(torch.load(models_name[model_name]))
    else:
        netSeg.apply(initialize_weights)

    return netSeg

# ----------------------------------------------------------------------------------------------------------------------
def save_print(model, path, mname):
   """ Save model architecture and number of parameters. """

   with open(os.path.join(path, mname + '.txt'), 'w') as f:
       print(model, file=f)
       print('', file=f)
       for i in range(3):
           print('###########################################################################################', file=f)
       print('', file=f)
       print("Parameter Count: %d" % count_parameters(model), file=f)

# ----------------------------------------------------------------------------------------------------------------------
def count_parameters(model):
   """ Count the number of parameters in the model. """

   return sum(p.numel() for p in model.parameters() if p.requires_grad)
