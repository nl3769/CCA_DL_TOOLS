def get_param_wandb(p, add_config):

    # --- convert class attributes to a dictionnary
    param4wandb = p.__dict__.copy()

    # --- select only interesting parameters
    rm_keys_PATH        = [key for key in param4wandb.keys() if "PATH" in key]
    rm_keys_WORKERS     = [key for key in param4wandb.keys() if "WORKERS" in key]
    rm_DATASET          = [key for key in param4wandb.keys() if "DATABASE" in key]
    rm_PDATA            = [key for key in param4wandb.keys() if "PDATA" in key]

    rm_keys = rm_keys_PATH + rm_keys_WORKERS + rm_DATASET + rm_PDATA

    # --- remove specific keys
    for key in rm_keys:
        del param4wandb[key]

    # --- convert values in string
    for key in param4wandb.keys():
        param4wandb[key] = str(param4wandb[key])

    param4wandb.update(add_config)

    return param4wandb