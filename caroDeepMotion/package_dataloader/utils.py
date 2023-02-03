from torch.utils.data                       import DataLoader
from package_dataloader.dataHandler         import dataHandler
from package_dataloader.dataHandlerFlow     import dataHandlerFlowFlyingChair
import package_utils.loader                 as pul
import numpy                                as np

def fetch_dataloader_flow(p):

    if p.SYNTHETIC_DATASET:
        trn_dataloader = dataHandler(p, "training")
        val_dataloader = dataHandler(p, "validation")
    else:
        trn_dataloader = dataHandlerFlowFlyingChair(p, "training")
        val_dataloader = dataHandlerFlowFlyingChair(p, "validation")
    args_training = {
        "dataset":      trn_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  p.WORKERS,
        "drop_last":    True}
    args_validation = {
        "dataset":      val_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  1,
        "drop_last":    True}
    if p.SYNTHETIC_DATASET:
        loader_training = DataLoader(**args_training)
        loader_validation = DataLoader(**args_validation)
    else:
        loader_training = DataLoader(**args_training)
        loader_validation = None

    return loader_training, loader_validation

# ----------------------------------------------------------------------------------------------------------------------
def fetch_dataloader(p):
    """ Create the data loader for the corresponding training set """

    trn_dataloader = dataHandler(p, "training")
    val_dataloader = dataHandler(p, "validation")
    test_dataloader = dataHandler(p, "testing")
    args_training = {
        "dataset":      trn_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  p.WORKERS,
        "drop_last":    True
        }
    args_validation = {
        "dataset":      val_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  1,
        "drop_last":    True
        }
    args_testing = {
        "dataset":      test_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      False,
        "num_workers":  1,
        "drop_last":    True
        }
    loader_training = DataLoader(**args_training)
    loader_validation = DataLoader(**args_validation)
    loader_testing = DataLoader(**args_testing)

    return loader_training, loader_validation, loader_testing

# ----------------------------------------------------------------------------------------------------------------------
def load_OF(path):

    OF = pul.load_nii(path)
    OF = np.array(OF)

    return OF

# ----------------------------------------------------------------------------------------------------------------------