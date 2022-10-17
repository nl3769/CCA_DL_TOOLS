import os
import torch

from torch                                          import nn
from package_network.network_dilatedUnet            import dilatedUnet

# ----------------------------------------------------------------------------------------------------------------------
def initialize_weights(m):
    """ Weights intialization.
    Args:
        TODO
    Returns:
        TODO
    """

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
    """ Get models name.
    Args:
        TODO
    Returns:
        TODO
    """

    models_name = {}
    if os.path.exists(pres):
        models = os.listdir(pres)

        if len(models)>0:
            for key in keys:
                for model in models:
                    if model.find(key) != -1 and model.find("val") != -1:
                        models_name[key] = os.path.join(pres, model)
        else:
            models_name = None
    else:
        models_name = None

    return models_name

# ----------------------------------------------------------------------------------------------------------------------
def load_model_flow(param):
    """ Load models with associated weights.
    Args:
        TODO
    Returns:
        TODO
    """

    # ---------------------
    # ---- LOAD MODELS ----
    # ---------------------

    netEncoder = NetEncoder(param)
    if param.MODEL_NAME == 'raft':
        netFlow = RAFT_NetFlow(param)
    elif param.MODEL_NAME == 'gma':
        netFlow = GMA_NetFlow(param)

    # -----------------------
    # ---- ADAPT WEIGHTS ----
    # -----------------------

    keys = ['netEncoder', 'netFlow']
    models_name = get_path_models(param.PRES, keys)

    if param.RESTORE_CHECKPOINT:
        if models_name is not None:
            for model_name in models_name.keys():
                if model_name == keys[0]:
                    netEncoder.load_state_dict(torch.load(models_name[model_name]))
                elif model_name == keys[2]:
                    netFlow.load_state_dict(torch.load(models_name[model_name]))
    else:
        netEncoder.apply(initialize_weights)
        netFlow.apply(initialize_weights)

    return netEncoder, netFlow

# ----------------------------------------------------------------------------------------------------------------------
def load_model(param):
    """ Load models with associated weights.
    Args:
        TODO
    Returns:
        TODO
    """

    # ---- load models
    netEncoder  = NetEncoder(param)
    netSeg      = dilatedUnet(
        input_nc        = 2,
        output_nc       = 2,
        n_layers        = param.NB_LAYERS,
        ngf             = param.NGF,
        kernel_size     = param.KERNEL_SIZE,
        padding         = param.PADDING,
        use_bias        = param.USE_BIAS
        )
    if param.MODEL_NAME == 'raft': 
        netFlow = RAFT_NetFlow(param)
    elif param.MODEL_NAME == 'gma':
        netFlow = GMA_NetFlow(param)

    # --- adapth weights
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
    """ Load models with associated weights.

    Args:
        param (metaclass): contains all project parameters.

    Returns:
        netSeg (metaclass): network for segmentation.
    """

    netSeg = dilatedUnet(
        input_nc        = 1,
        output_nc       = 1,
        n_layers        = param.NB_LAYERS,
        ngf             = param.NGF,
        kernel_size     = param.KERNEL_SIZE,
        padding         = param.PADDING,
        use_bias        = param.USE_BIAS,
        dropout         = param.DROPOUT
    )

    # --- adapt weights
    keys = ['netSeg']
    models_name = get_path_models(param.PATH_SAVE_MODEL, keys)

    # --- if exist, load pretrained weights of the model
    if param.RESTORE_CHECKPOINT and models_name != None:
        netSeg.load_state_dict(torch.load(models_name[keys[0]]))
    else:
        netSeg.apply(initialize_weights)

    return netSeg

# ----------------------------------------------------------------------------------------------------------------------
def save_print(model, path, mname):
   """ Save model architecture and number of parameters.
        Args:
            TODO
        Returns:
            TODO
   """

   with open(os.path.join(path, mname + '.txt'), 'w') as f:
       print(model, file=f)
       print('', file=f)
       for i in range(3):
           print('###########################################################################################', file=f)
       print('', file=f)
       print("Parameter Count: %d" % count_parameters(model), file=f)

# ----------------------------------------------------------------------------------------------------------------------
def count_parameters(model):
   """ Count the number of parameters in the model.
       Args:
        TODO
    Returns:
        TODO
   """

   return sum(p.numel() for p in model.parameters() if p.requires_grad)
