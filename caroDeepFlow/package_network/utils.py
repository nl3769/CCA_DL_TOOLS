import os
import torch

from torch                                          import nn

from package_network.netFlow                        import NetFlow
from package_network.netEncoder                     import NetEncoder
from package_network.netSegDecoder                  import NetSegDecoder
from package_network.dilatedUnet                    import DilatedUnet

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

    if param.FEATURES == "shared":

        netEncoder  = NetEncoder(param)
        netSeg      = NetSegDecoder(param)
        netFlow     = NetFlow(param)

        return netEncoder, netSeg, netFlow

    elif param.FEATURES == 'split':

        netEncoder  = NetEncoder(param)
        netFlow     = NetFlow(param)
        netSeg      = DilatedUnet(in_channels=2, out_channels=2)


    # -----------------------
    # ---- ADAPT WEIGHTS ----
    # -----------------------

    keys = ['netEncoder', 'netSeg', 'netFlow']
    models_name = get_path_models(param.PRES, keys)

    if param.RESTORE_CHECKPOINT and models_name != None:

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
