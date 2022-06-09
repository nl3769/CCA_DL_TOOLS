'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import torch
import numpy                                as np
import package_utils.pytorch_processing     as pupp

from tqdm                                   import tqdm

# ----------------------------------------------------------------------------------------------------------------------
def training_loop_split(param, networks, segLoss, flowLoss, optimizers, scheduler, logger, loader, id_epoch, device, IMCExtractor):

    seg_loss_ = []
    seg_metrics_ = {}
    for key in segLoss.metrics:
        seg_metrics_[key] = []

    flow_loss_ = []
    flow_metrics_ = {}
    for key in flowLoss.metrics:
        flow_metrics_[key] = []

    full_loss = []

    for i_batch, (I1, I2, M1, M2, OF, CF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'TRAINING - Epoch id.: {id_epoch}')):

        # --- load data
        I1, I2, M1, M2, OF = I1.to(device), I2.to(device), M1.to(device), M2.to(device), OF.to(device)
        IMCExtractor.update(M1, CF['yCF'])
        M1_c, _, _ = IMCExtractor()
        M1_c = torch.tensor(M1_c).to(device)
        OF = torch.mul(OF, M1_c)

        ##########################
        # --- TRAIN NETWORKS --- #
        ##########################

        # --- set gradient to zero
        optimizers["netEncoder"].zero_grad()
        optimizers["netSeg"].zero_grad()
        optimizers["netFlow"].zero_grad()

        # --- set model in training mode
        networks["netEncoder"].train()
        networks["netSeg"].train()
        networks["netFlow"].train()

        fmap1, skc1, fmap2, skc2 = networks["netEncoder"](I1, I2)
        M1_pred, M2_pred = networks["netSeg"](I1, I2)
        cp_M1 = torch.clone(M1_pred).detach()
        cp_M1 = pupp.treshold_mask(cp_M1)
        IMCExtractor.update(cp_M1, CF['yCF'])
        M1_c, _, IMT1 = IMCExtractor()
        M1_c = torch.tensor(M1_c).to(device)
        flow_pred = networks["netFlow"](I1, fmap1, fmap2, M1_c)

        flow_loss, flow_metrics = flowLoss(flow_pred, OF, param.GAMMA)
        seg_loss, seg_metrics = segLoss(M1_pred, M2_pred, M1, M2)
        loss = flow_loss + seg_loss

        # loss.backward()
        flow_loss.backward()
        seg_loss.backward()

        optimizers["netEncoder"].step()
        optimizers["netSeg"].step()
        optimizers["netFlow"].step()

        scheduler["netEncoder"].step()
        scheduler["netSeg"].step()
        scheduler["netFlow"].step()

        ##################
        # --- LOGGER --- #
        ##################

        # --- segmentation
        seg_loss_.append(seg_loss.cpu().detach().numpy())
        [seg_metrics_[key].append(seg_metrics[key].cpu().detach().numpy()) for key in seg_metrics.keys()]
        # --- flow
        flow_loss_.append(flow_loss.cpu().detach().numpy())
        [flow_metrics_[key].append(flow_metrics[key]) for key in flow_metrics.keys()]
        # --- full loss
        full_loss.append(loss.cpu().detach().numpy())

    ###############################
    # --- SAVE LOSSES/METRICS --- #
    ###############################

    # --- segmentation branch
    seg_loss_ = np.array(seg_loss_)
    seg_loss_ = np.mean(seg_loss_)
    for key in seg_metrics_.keys():
        seg_metrics_[key] = np.mean(np.array(seg_metrics_[key]))
    logger.add_loss_seg(seg_loss_,       set='training')
    logger.add_metrics_seg(seg_metrics_, set='training')

    # --- flow branch
    flow_loss_ = np.array(flow_loss_)
    flow_loss_ = np.mean(flow_loss_)
    for key in flow_metrics_.keys():
        flow_metrics_[key] = np.mean(np.array(flow_metrics_[key]))

    # --- full loss
    full_loss = np.array(full_loss)
    full_loss = np.mean(full_loss)

    # --- update
    logger.add_loss_flow(flow_loss_,        set='training')
    logger.add_metrics_flow(flow_metrics_,  set='training')
    logger.add_loss_full(full_loss,         set='training')

# ----------------------------------------------------------------------------------------------------------------------
def training_loop_shared(param, networks, segLoss, flowLoss, optimizers, scheduler, logger, loader, id_epoch, device, IMCExtractor):

    seg_loss_ = []
    seg_metrics_ = {}
    for key in segLoss.metrics:
        seg_metrics_[key] = []

    flow_loss_ = []
    flow_metrics_ = {}
    for key in flowLoss.metrics:
        flow_metrics_[key] = []

    full_loss = []

    for i_batch, (I1, I2, M1, M2, OF, CF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'TRAINING - Epoch id.: {id_epoch}')):

        # --- load data
        I1, I2, M1, M2, OF = I1.to(device), I2.to(device), M1.to(device), M2.to(device), OF.to(device)
        IMCExtractor.update(M1, CF['yCF'])
        M1_c, _, _ = IMCExtractor()
        M1_c = torch.tensor(M1_c).to(device)
        OF = torch.mul(OF, M1_c)

        ##########################
        # --- TRAIN NETWORKS --- #
        ##########################

        # --- set gradient to zero
        optimizers["netEncoder"].zero_grad()
        optimizers["netSeg"].zero_grad()
        optimizers["netFlow"].zero_grad()

        # --- set model in training mode
        networks["netEncoder"].train()
        networks["netSeg"].train()
        networks["netFlow"].train()

        fmap1, skc1, fmap2, skc2 = networks["netEncoder"](I1, I2)
        M1_pred, M2_pred = networks["netSeg"](fmap1, skc1, fmap2, skc2)
        cp_M1 = torch.clone(M1_pred).detach()
        cp_M1 = pupp.treshold_mask(cp_M1)
        IMCExtractor.update(cp_M1, CF['yCF'])
        M1_c, _, IMT1 = IMCExtractor()
        M1_c = torch.tensor(M1_c).to(device)
        flow_pred = networks["netFlow"](I1, fmap1, fmap2, M1_c)

        flow_loss, flow_metrics = flowLoss(flow_pred, OF, param.GAMMA)
        seg_loss, seg_metrics = segLoss(M1_pred, M2_pred, M1, M2)
        loss = flow_loss + seg_loss

        loss.backward()

        optimizers["netEncoder"].step()
        optimizers["netSeg"].step()
        optimizers["netFlow"].step()

        scheduler["netEncoder"].step()
        scheduler["netSeg"].step()
        scheduler["netFlow"].step()

        ##################
        # --- LOGGER --- #
        ##################

        # --- segmentation
        seg_loss_.append(seg_loss.cpu().detach().numpy())
        [seg_metrics_[key].append(seg_metrics[key].cpu().detach().numpy()) for key in seg_metrics.keys()]
        # --- flow
        flow_loss_.append(flow_loss.cpu().detach().numpy())
        [flow_metrics_[key].append(flow_metrics[key]) for key in flow_metrics.keys()]
        # --- full loss
        full_loss.append(loss.cpu().detach().numpy())

    ###############################
    # --- SAVE LOSSES/METRICS --- #
    ###############################

    # --- segmentation branch
    seg_loss_ = np.array(seg_loss_)
    seg_loss_ = np.mean(seg_loss_)
    for key in seg_metrics_.keys():
        seg_metrics_[key] = np.mean(np.array(seg_metrics_[key]))
    logger.add_loss_seg(seg_loss_,       set='training')
    logger.add_metrics_seg(seg_metrics_, set='training')

    # --- flow branch
    flow_loss_ = np.array(flow_loss_)
    flow_loss_ = np.mean(flow_loss_)
    for key in flow_metrics_.keys():
        flow_metrics_[key] = np.mean(np.array(flow_metrics_[key]))

    # --- full loss
    full_loss = np.array(full_loss)
    full_loss = np.mean(full_loss)

    # --- update
    logger.add_loss_flow(flow_loss_,        set='training')
    logger.add_metrics_flow(flow_metrics_,  set='training')
    logger.add_loss_full(full_loss,         set='training')

# ----------------------------------------------------------------------------------------------------------------------