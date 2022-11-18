'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch

import numpy                                as np
import package_utils.pytorch_processing     as pupp

from tqdm                                   import tqdm

# ----------------------------------------------------------------------------------------------------------------------
def val_loop_seg(param, networks, segLoss, logger, loader, id_epoch, device):

    seg_loss_ = []
    seg_metrics_ = {}
    for key in segLoss.metrics:
        seg_metrics_[key] = []

    # --- set model in training mode
    networks["netSeg"].eval()
    
    save = True

    for i_batch, (I1, M1, CF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'VAL - Epoch id.: {id_epoch}')):

        # --- load data
        I1, M1 = I1.to(device), M1.to(device)

        # --- inference
        M1_pred = networks["netSeg"](I1)

        # --- loss
        seg_loss, seg_metrics = segLoss(M1_pred, M1)
        seg_loss_.append(seg_loss.cpu().detach().numpy())
        [seg_metrics_[key].append(seg_metrics[key].cpu().detach().numpy()) for key in seg_metrics.keys()]
        if save:
            # --- save one random image
            I1 = I1[0, ].cpu().detach().numpy().squeeze()
            M1 = M1[0, ].cpu().detach().numpy().squeeze()
            M1_pred = M1_pred[0, ].cpu().detach().numpy().squeeze()
            logger.plot1Seg(I1, M1, M1_pred, param.PATH_RANDOM_PRED_TRN, 'val_' + str(id_epoch) + "_" + fname[0])
            save = False
    ##################
    # --- LOGGER --- #
    ##################

    seg_loss_ = np.array(seg_loss_)
    seg_loss_ = np.mean(seg_loss_)
    for key in seg_metrics_.keys():
        seg_metrics_[key] = np.mean(np.array(seg_metrics_[key]))
    logger.add_loss_seg(seg_loss_,       set='validation')
    logger.add_metrics_seg(seg_metrics_, set='validation')

    return seg_loss_, seg_metrics_