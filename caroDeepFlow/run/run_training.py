'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import argparse
import importlib
import torch
import math
import random
import wandb

import torch.optim						    as optim
import numpy							    as np
import package_dataloader.utils 		    as pdlu
import package_network.network 			    as pnn
import package_loss.lossSeg 			    as plls
import package_loss.lossFlow 			    as plll
import package_logger.logger 			    as plog
import package_utils.pytorch_processing     as pupp
from tqdm 								    import tqdm

from icecream 							    import ic
import package_debug.visualisation 		    as pdv

################################
# --- RUN TRAINING ROUTINE --- #
################################

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    ic(device)

    # --- get dataloader
    training_dataloader, validation_dataloader, _ = pdlu.fetch_dataloader(p)

    # --- load models
    netEncoder = pnn.NetEncoder(p)
    netDecoder = pnn.NetSegDecoder(p)
    netFlow = pnn.NetFlow(p)
    netEncoder = netEncoder.to(device)
    netDecoder = netDecoder.to(device)
    netFlow = netFlow.to(device)

    # --- optimizer/scheduler
    optimizerEncoder, schedulerEncoder = fetch_optimizer(p, netEncoder, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)
    optimizerDecoder, schedulerDecoder = fetch_optimizer(p, netDecoder, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)
    optimizerFlow, schedulerFlow = fetch_optimizer(p, netFlow, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)

    # --- losses
    segLoss = plls.lossSeg()
    flowLoss = plll.lossFlow()

    # --- store optimizers/schedulers and optimizers
    networks = {"netEncoder": netEncoder,
                "netDecoder": netDecoder,
                "netFlow": netFlow}
    optimizers = {"netEncoder": optimizerEncoder,
                  "netDecoder": optimizerDecoder,
                  "netFlow": optimizerFlow}
    schedulers = {"netEncoder": schedulerEncoder,
                 "netDecoder": schedulerDecoder,
                 "netFlow": schedulerFlow}

    # --- logger
    logger = plog.loggerClass(p, segLoss.metrics.keys(), flowLoss.metrics.keys())

    # --- loop
    for epoch in range(p.NB_EPOCH):

        training_loop(p, networks, segLoss, flowLoss, optimizers, schedulers, logger, training_dataloader, epoch, device)
        validation_loop(p, networks, segLoss, flowLoss, logger, validation_dataloader, epoch, device)
        logger.save_best_model(epoch, networks)

    logger.plot_loss()
    logger.plot_metrics()
    logger.save_model_history()

# ----------------------------------------------------------------------------------------------------------------------
def config_wandb():
    config = {}
    project_name = {}

    return config, project_name
# ----------------------------------------------------------------------------------------------------------------------
def training_loop(param, networks, segLoss, flowLoss, optimizers, scheduler, logger, loader, id_epoch, device):


    seg_loss_ = []
    seg_metrics_ = {}
    for key in segLoss.metrics:
        seg_metrics_[key] = []

    flow_loss_ = []
    flow_metrics_ = {}
    for key in flowLoss.metrics:
        flow_metrics_[key] = []

    full_loss = []

    for i_batch, (I1, I2, M1, M2, OF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'TRAINING - Epoch id.: {id_epoch}')):

        # --- load data
        I1, I2, M1, M2, OF = I1.to(device), I2.to(device), M1.to(device), M2.to(device), OF.to(device)

        ##########################
        # --- TRAIN NETWORKS --- #
        ##########################

        # --- set gradient to zero
        optimizers["netEncoder"].zero_grad()
        optimizers["netDecoder"].zero_grad()
        optimizers["netFlow"].zero_grad()

        networks["netEncoder"].train()
        networks["netDecoder"].train()
        networks["netFlow"].train()

        fmap1, skc1, fmap2, skc2 = networks["netEncoder"](I1, I2)
        M1_pred, M2_pred = networks["netDecoder"](fmap1, skc1, fmap2, skc2)
        cp_M1 = torch.clone(M1_pred).detach()
        cp_M1 = pupp.treshold_mask(cp_M1, 0.5)
        flow_pred = networks["netFlow"](I1, fmap1, fmap2, cp_M1)

        flow_loss, flow_metrics = flowLoss(flow_pred, OF, param.GAMMA)
        seg_loss, seg_metrics = segLoss(M1_pred, M2_pred, M1, M2)
        loss = flow_loss + seg_loss

        loss.backward()
        optimizers["netEncoder"].step()
        optimizers["netDecoder"].step()
        optimizers["netFlow"].step()

        scheduler["netEncoder"].step()
        scheduler["netDecoder"].step()
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
    logger.add_loss_seg(seg_loss_, set='training')
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
    logger.add_loss_flow(flow_loss_, set='training')
    logger.add_metrics_flow(flow_metrics_, set='training')
    logger.add_loss_full(full_loss, set='training')

# ----------------------------------------------------------------------------------------------------------------------
def validation_loop(param, networks, segLoss, flowLoss, logger, loader, id_epoch, device):

    # --- segmentation loss
    seg_loss_ = []
    seg_metrics_ = {}
    for key in segLoss.metrics:
        seg_metrics_[key] = []

    # --- optical flow loss
    flow_loss_ = []
    flow_metrics_ = {}
    for key in flowLoss.metrics:
        flow_metrics_[key] = []

    # --- full loss
    full_loss = []

    save = True

    for i_batch, (I1, I2, M1, M2, OF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'VALIDATION - Epoch id.: {id_epoch}')):

        # --- load data
        I1, I2, M1, M2, OF = I1.to(device), I2.to(device), M1.to(device), M2.to(device), OF.to(device)
        M1 = pupp.treshold_mask(M1)
        OF = torch.mul(OF, M1)
        networks["netEncoder"].eval()
        networks["netDecoder"].eval()
        networks["netFlow"].eval()

        ######################
        # --- PREDICTION --- #
        ######################

        fmap1, skc1, fmap2, skc2 = networks["netEncoder"](I1, I2)
        M1_pred, M2_pred = networks["netDecoder"](fmap1, skc1, fmap2, skc2)
        flow_pred = networks["netFlow"](I1, fmap1, fmap2, M1_pred)

        seg_loss, seg_metrics = segLoss(M1_pred, M2_pred, M1, M2)
        flow_loss, flow_metrics = flowLoss(flow_pred, OF, param.GAMMA)
        loss = seg_loss + flow_loss

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

        if save == True:

            mask1_gt, mask2_gt = M1.cpu().detach().numpy(), M2.cpu().detach().numpy()
            mask1_pred, mask2_pred = M1_pred.cpu().detach().numpy(), M2_pred.cpu().detach().numpy()
            OF_pred = flow_pred[-1].cpu().detach().numpy()
            OF_gt = OF.cpu().detach().numpy()
            mask1_gt, mask2_gt = mask1_gt[0, 0, ], mask2_gt[0, 0, ]
            mask1_pred, mask2_pred = mask1_pred[0, 0,], mask2_pred[0, 0,]
            OF_pred, OF_gt = OF_pred[0, ], OF_gt[0, ]
            I1_, I2_ = I1.cpu().detach().numpy(), I2.cpu().detach().numpy()
            I1_, I2_ = I1_[0, 0, ], I2_[0, 0, ]
            logger.plot_pred(I1_, I2_, mask1_gt, mask2_gt, mask1_pred, mask2_pred, OF_gt, OF_pred, id_epoch, fname[0])
            save = False

    #############################
    # --- SAVE LOSS/METRICS --- #
    #############################

    # --- segmentation branch
    seg_loss_ = np.array(seg_loss_)
    seg_loss_ = np.mean(seg_loss_)
    for key in seg_metrics_.keys():
        seg_metrics_[key] = np.mean(np.array(seg_metrics_[key]))
    logger.add_loss_seg(seg_loss_, set='validation')
    logger.add_metrics_seg(seg_metrics_, set='validation')

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

# ----------------------------------------------------------------------------------------------------------------------
def fetch_optimizer(p, model, n_step):
    """ Create the optimizer and learning rate scheduler. """

    # --- optimizer
    optimizer = optim.Adam(model.parameters(), lr=p.LEARNING_RATE)

    # --- schedular
    param = {'max_lr': p.LEARNING_RATE,
             'total_steps': n_step,
             'epochs': p.NB_EPOCH,
             'pct_start': 0.05,
             'cycle_momentum': False,
             'anneal_strategy': 'linear'}
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **param)

    return optimizer, scheduler

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
