'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch
import numpy                                as np
import package_utils.pytorch_processing     as pupp

from tqdm                                   import tqdm

# ----------------------------------------------------------------------------------------------------------------------
def trn_loop_seg(param, networks, segLoss, optimizers, scheduler, logger, loader, id_epoch, device):
    """ Training loop."""

    # --- init.  variables
    seg_loss_ = []
    seg_metrics_ = {}
    for key in segLoss.metrics:
        seg_metrics_[key] = []
    # --- set model in training mode
    networks["netSeg"].train()

    save = True
    # --- loop
    for i_batch, (I1, M1, CF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'TRN - Epoch id.: {id_epoch}')):

        # --- load data
        I1, M1 = I1.to(device), M1.to(device)

        ##########################
        # --- TRAIN NETWORKS --- #
        ##########################

        optimizers["netSeg"].zero_grad()
        M1_pred = networks["netSeg"](I1)
        seg_loss, seg_metrics = segLoss(M1_pred, M1)
        seg_loss.backward()
        optimizers["netSeg"].step()
        scheduler["netSeg"].step()

        ##################
        # --- LOGGER --- #
        ##################

        seg_loss_.append(seg_loss.cpu().detach().numpy())
        [seg_metrics_[key].append(seg_metrics[key].cpu().detach().numpy()) for key in seg_metrics.keys()]

        if save == True:
            # --- save one random image
            I1 = I1[0, ].cpu().detach().numpy().squeeze()
            M1 = M1[0, ].cpu().detach().numpy().squeeze()
            M1_pred = M1_pred[0, ].cpu().detach().numpy().squeeze()
            logger.plot1Seg(I1, M1, M1_pred, param.PATH_RANDOM_PRED_TRN, 'trn_' + str(id_epoch) + "_" + fname[0])
            save = False

    ###############################
    # --- SAVE LOSSES/METRICS --- #
    ###############################

    # --- segmentation branch
    seg_loss_ = np.array(seg_loss_)
    seg_loss_ = np.mean(seg_loss_)
    for key in seg_metrics_.keys():
        seg_metrics_[key] = np.mean(np.array(seg_metrics_[key]))
    logger.add_loss_seg(seg_loss_, set='training')
    logger.add_metrics_seg(seg_metrics_, set='training')

    return seg_loss_, seg_metrics_
# ----------------------------------------------------------------------------------------------------------------------
