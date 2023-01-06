# ----------------------------------------------------------------------------------------------------------------------
def get_param_wandb(p):

    # --- convert class attributes to a dictionnary
    param4wandb = p.__dict__.copy()

    # --- select only interesting parameters
    rm_keys = {}
    rm_keys['rm_keys_PATH'] = [key for key in param4wandb.keys() if "PDATA" in key]
    rm_keys['rm_keys_DEVICE'] = [key for key in param4wandb.keys() if "PSPLIT" in key]
    rm_keys['rm_keys_PRES'] = [key for key in param4wandb.keys() if "PRES" in key]
    rm_keys['rm_keys_GPU'] = [key for key in param4wandb.keys() if "DEVICE" in key]
    rm_keys['rm_keys_CUDA'] = [key for key in param4wandb.keys() if "CUDA" in key]
    rm_keys['rm_keys_WORKERS'] = [key for key in param4wandb.keys() if "WORKERS" in key]
    rm_keys['rm_EXPERIMENT_NAME'] = [key for key in param4wandb.keys() if "EXPNAME" in key]
    rm_keys['rm_ENTITY'] = [key for key in param4wandb.keys() if "ENTITY" in key]
    rm_keys['rm_PATH'] = [key for key in param4wandb.keys() if "PATH" in key]
    rm_keys['rm_USER'] = [key for key in param4wandb.keys() if "USER" in key]


    # --- remove specific keys
    for key in rm_keys.keys():
        for rm in rm_keys[key]:
            del param4wandb[rm]

    # --- convert values in string
    for key in param4wandb.keys():
        param4wandb[key] = str(param4wandb[key])

    return param4wandb

# ----------------------------------------------------------------------------------------------------------------------