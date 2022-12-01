from torch.utils.data                       import DataLoader
from package_dataloader.dataHandlerSeg      import dataHandlerSegInSilico, dataHandlerSegCubs
import package_utils.loader                 as pul
import numpy                                as np

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


    loader_trn = DataLoader(**args_trn)
    loader_val = DataLoader(**args_val)
    loader_tst = DataLoader(**args_tst)

    return loader_trn, loader_val, loader_tst

# ----------------------------------------------------------------------------------------------------------------------