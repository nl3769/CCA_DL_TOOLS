from torch.utils.data                       import DataLoader
from package_dataloader.dataHandler         import dataHandler
from package_dataloader.dataHandlerFlow     import dataHandlerFlowFlyingChair
from package_dataloader.dataHandlerSeg      import dataHandlerSegInSilico, dataHandlerSegCubs
import package_utils.loader                 as pul
import numpy                                as np

def fetch_dataloader_flow(p):
    # TODO: specify database if we want to train in more than flyingchairs
    trn_dataloader = dataHandlerFlowFlyingChair(p, "training")
    val_dataloader = dataHandlerFlowFlyingChair(p, "validation")

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

    loader_training     = DataLoader(**args_training)
    loader_validation   = DataLoader(**args_validation)

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
def fetch_dataloader_seg(p):

    # --- pretrained on in silico data
    if p.IN_VIVO:
        trn_dataloader = dataHandlerSegCubs(p, "training")
        val_dataloader = dataHandlerSegCubs(p, "validation")
        tst_dataloader = dataHandlerSegCubs(p, "testing")
    else:
        trn_dataloader = dataHandlerSegInSilico(p, "training")
        val_dataloader = dataHandlerSegInSilico(p, "validation")
        tst_dataloader = dataHandlerSegInSilico(p, "testing")

    args_trn = {
        "dataset":      trn_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  p.WORKERS,
        "drop_last":    True
        }
    args_val = {
        "dataset":      val_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  1,
        "drop_last":    True
        }
    args_tst = {
        "dataset":      tst_dataloader,
        "batch_size":   p.BATCH_SIZE,
        "pin_memory":   False,
        "shuffle":      True,
        "num_workers":  1,
        "drop_last":    True
        }


    loader_trn     = DataLoader(**args_trn)
    loader_val   = DataLoader(**args_val)
    loader_tst = DataLoader(**args_tst)

    return loader_trn, loader_val, loader_tst

# ----------------------------------------------------------------------------------------------------------------------
def load_OF(path):

    OF = pul.load_nii(path)
    OF = np.array(OF)

    return OF

# ----------------------------------------------------------------------------------------------------------------------
