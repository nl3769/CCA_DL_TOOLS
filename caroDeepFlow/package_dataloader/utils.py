from torch.utils.data                       import DataLoader
from package_dataloader.dataHandler         import dataHandler
from package_dataloader.dataHandlerSeg      import dataHandlerSeg
import package_utils.loader                 as pul
import numpy                                as np

# ----------------------------------------------------------------------------------------------------------------------
def fetch_dataloader(param):
    """ Create the data loader for the corresponding training set """

    training_dataloader     = dataHandler(param, "training")
    validation_dataloader   = dataHandler(param, "validation")
    testing_dataloader      = dataHandler(param, "testing")

    args_training = {
        "dataset":      training_dataloader,
        "batch_size":   param.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  param.WORKERS,
        "drop_last":    True}
    args_validation = {
        "dataset":      validation_dataloader,
        "batch_size":   param.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  1,
        "drop_last":    True}
    args_testing = {
        "dataset":      testing_dataloader,
        "batch_size":   param.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      False,
        "num_workers":  1,
        "drop_last":    True}

    loader_training     = DataLoader(**args_training)
    loader_validation   = DataLoader(**args_validation)
    loader_testing      = DataLoader(**args_testing)

    return loader_training, loader_validation, loader_testing

# ----------------------------------------------------------------------------------------------------------------------
def fetch_dataloader_seg(param):
    """ Create the data loader for the corresponding training set """

    training_dataloader     = dataHandlerSeg(param, "training")
    validation_dataloader   = dataHandlerSeg(param, "validation")
    testing_dataloader      = dataHandlerSeg(param, "testing")

    args_training = {
        "dataset":         training_dataloader,
        "batch_size":      param.BATCH_SIZE,
        "pin_memory":      False,
        "shuffle":         True,
        "num_workers":     param.WORKERS,
        "drop_last":       True}
    args_validation = {
        "dataset":       validation_dataloader,
        "batch_size":    param.BATCH_SIZE,
        "pin_memory":    False,
        "shuffle":       True,
        "num_workers":   1,
        "drop_last":     True}
    args_testing = {
        "dataset":          testing_dataloader,
        "batch_size":    param.BATCH_SIZE,
        "pin_memory":    False,
        "shuffle":       False,
        "num_workers":   1,
        "drop_last":     True}

    loader_training     = DataLoader(**args_training)
    loader_validation   = DataLoader(**args_validation)
    loader_testing      = DataLoader(**args_testing)

    return loader_training, loader_validation, loader_testing

# ----------------------------------------------------------------------------------------------------------------------
def load_OF(path):

    OF = pul.load_nii(path)
    OF = np.array(OF)

    return OF

# ----------------------------------------------------------------------------------------------------------------------