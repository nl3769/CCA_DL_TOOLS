'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch

import numpy                                as np
import package_utils.pytorch_processing     as pupp

from tqdm                                   import tqdm

# ----------------------------------------------------------------------------------------------------------------------
def val_loop_flow(param, networks, flowLoss, logger, loader, id_epoch, device):
    # --- displacement field loss and full loss

    flow_loss_ = []
    full_loss = []
    flow_metrics_ = {}
    for key in flowLoss.metrics:
        flow_metrics_[key] = []

    save = True
    networks["netEncoder"].eval()
    networks["netFlow"].eval()

    for i_batch, (I1, I2, OF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'VALIDATION - Epoch id.: {id_epoch}')):

        # --- load data
        I1, I2, OF = I1.to(device).float(), I2.to(device).float(), OF.to(device).float()

        ######################
        # --- PREDICTION --- #
        ######################

        fmap1, skc1, fmap2, _ = networks["netEncoder"](I1, I2)
        mask = torch.ones(I1.shape).float().to(device)
        flow_pred = networks["netFlow"](I1, fmap1, fmap2, mask)

        flow_loss, flow_metrics = flowLoss(flow_pred, OF, param.GAMMA)

        ##################
        # --- LOGGER --- #
        ##################

        # --- flow
        print(flow_loss)
        flow_loss_.append(flow_loss.cpu().detach().numpy())
        [flow_metrics_[key].append(flow_metrics[key]) for key in flow_metrics.keys()]
        # --- full loss
        full_loss.append(flow_loss.cpu().detach().numpy())

        if save == True:

            OF_pred = flow_pred[-1].cpu().detach().numpy()
            OF_gt = OF.cpu().detach().numpy()
            OF_pred, OF_gt = OF_pred[0, ], OF_gt[0, ]
            I1_, I2_ = I1.cpu().detach().numpy(), I2.cpu().detach().numpy()
            I1_, I2_ = I1_[0, 0, ], I2_[0, 0, ]
            logger.plot_pred_flow(I1_, I2_, OF_gt, OF_pred, id_epoch, param.PATH_SAVE_PRED_TRAINING, "val" , fname[0])
            save = False

    #############################
    # --- SAVE LOSS/METRICS --- #
    #############################

    # --- flow branch
    flow_loss_ = np.array(flow_loss_)
    flow_loss_ = np.mean(flow_loss_)
    for key in flow_metrics_.keys():
        flow_metrics_[key] = np.mean(np.array(flow_metrics_[key]))

    # --- full loss
    full_loss = np.array(full_loss)
    full_loss = np.mean(full_loss)

    # --- update
    logger.add_loss_flow(flow_loss_,        set='validation')
    logger.add_metrics_flow(flow_metrics_,  set='validation')
    logger.add_loss_full(full_loss,         set='validation')

# ----------------------------------------------------------------------------------------------------------------------
def val_loop_synth(param, networks, flowLoss, logger, loader, id_epoch, device):

    # --- displacement field loss and full loss
    flow_loss_ = []
    full_loss = []
    flow_metrics_ = {}

    for key in flowLoss.metrics:
        flow_metrics_[key] = []

    save = True
    networks["netEncoder"].eval()
    networks["netFlow"].eval()

    for i_batch, (I1, I2, M1, M2, OF, CF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'VALIDATION - Epoch id.: {id_epoch}')):

        # --- load data
        I1, I2, M1, M2, OF = I1.to(device).float(), I2.to(device).float(), M1.to(device).float(), M2.to(device).float(), OF.to(device).float()
        OF = torch.mul(OF, M1)
        ######################
        # --- PREDICTION --- #
        ######################

        fmap1, skc1, fmap2, _ = networks["netEncoder"](I1, I2)
        flow_pred = networks["netFlow"](I1, fmap1, fmap2, M1)
        flow_loss, flow_metrics = flowLoss(flow_pred, OF, param.GAMMA)

        ##################
        # --- LOGGER --- #
        ##################

        # --- flow
        flow_loss_.append(flow_loss.cpu().detach().numpy())
        [flow_metrics_[key].append(flow_metrics[key]) for key in flow_metrics.keys()]
        # --- full loss
        full_loss.append(flow_loss.cpu().detach().numpy())

        if save == True:

            OF_pred = flow_pred[-1].cpu().detach().numpy()
            OF_gt = OF.cpu().detach().numpy()
            OF_pred, OF_gt = OF_pred[0, ], OF_gt[0, ]
            I1_, I2_ = I1.cpu().detach().numpy(), I2.cpu().detach().numpy()
            I1_, I2_ = I1_[0, 0, ], I2_[0, 0, ]
            logger.plot_pred_flow(I1_, I2_, OF_gt, OF_pred, id_epoch, param.PATH_SAVE_PRED_TRAINING, "val", fname[0])
            logger.plot_pred_flow_error(OF_gt, OF_pred, id_epoch, param.PATH_SAVE_PRED_TRAINING, "val", fname[0])

            save = False

    #############################
    # --- SAVE LOSS/METRICS --- #
    #############################

    # --- flow branch
    flow_loss_ = np.array(flow_loss_)
    flow_loss_ = np.mean(flow_loss_)
    for key in flow_metrics_.keys():
        flow_metrics_[key] = np.mean(np.array(flow_metrics_[key]))

    # --- full loss
    full_loss = np.array(full_loss)
    full_loss = np.mean(full_loss)

    # --- update
    logger.add_loss_flow(flow_loss_, set='validation')
    logger.add_metrics_flow(flow_metrics_, set='validation')
    logger.add_loss_full(full_loss, set='validation')