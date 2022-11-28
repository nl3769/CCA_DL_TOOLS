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

import torch.optim				                  as optim
import package_dataloader.utils 		          as pdlu
import package_network.utils 			          as pnu
import package_loss.lossSeg 			          as plls
import package_logger.logger 			          as plog
import package_loop.trn_loop                      as trn
import package_loop.tst_loop                      as tst
import package_loop.val_loop                      as val
import package_utils.wandb_utils                  as puwu

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
    # --- get dataloader
    trn_dataloader, val_dataloader, tst_dataloader = pdlu.fetch_dataloader_seg(p)
    # --- load models
    netSeg = pnu.load_model_seg(p)
    netSeg = netSeg.to(device)
    # --- print model in txt file
    pnu.save_print(netSeg, p.PATH_PRINT_MODEL, 'netSeg')
    # --- optimizer/scheduler
    optimizer, scheduler = fetch_optimizer(p, netSeg)
    # --- loss
    segLoss = plls.lossSeg1Frame()
    # --- store optimizers/schedulers and optimizers
    networks = {"netSeg": netSeg}
    optimizers = {"netSeg": optimizer}
    schedulers = {"netSeg": scheduler}
    # --- logger
    logger = plog.loggerClassSeg(p, segLoss.metrics.keys())
    # --- config wandb
    if p.USE_WANDB:
        config = puwu.get_param_wandb(p)
        wandb.init(project="caroDeepSegPytorch", entity=p.ENTITY, dir=p.PATH_WANDB, config=config, name=p.EXPNAME)

    # --- trn/val loop
    for epoch in range(p.NB_EPOCH):
        loss_trn, metric_trn = trn.trn_loop_seg(p, networks, segLoss, optimizers, schedulers, logger, trn_dataloader, epoch, device)
        loss_val, metric_val = val.val_loop_seg(p, networks, segLoss, logger, val_dataloader, epoch, device)
        logger.save_best_model(epoch, networks)

        # --- Log information to wandb
        lr = get_lr(optimizer)
        if p.USE_WANDB:
            wandb.log({"loss_trn": loss_trn,
                       "trn_BCE": metric_trn['BCE_I1'],
                       "trn_DICE": metric_trn['dice_I1'],
                       "loss_val": loss_val,
                       "val_BCE": metric_trn['BCE_I1'],
                       "val_DICE": metric_trn['dice_I1'],
                       "learning_rate": lr})

        if logger.early_stop_id >= p.EARLY_STOP:
            print('EARLY STOP')
            break

    logger.plot_loss()
    logger.plot_metrics()
    logger.save_model_history()

    p.RESTORE_CHECKPOINT = True
    netSeg = pnu.load_model_seg(p)
    netSeg = netSeg.to(device)
    networks = {"netSeg": netSeg}
    tst.tst_loop_seg(p, networks, segLoss, val_dataloader, device, 'val')
    tst.tst_loop_seg(p, networks, segLoss, tst_dataloader, device, 'tst')
    tst.tst_loop_seg(p, networks, segLoss, trn_dataloader, device, 'trn')


# ----------------------------------------------------------------------------------------------------------------------
def fetch_optimizer(p, model):
    """ Create the optimizer and learning rate scheduler. """

    # --- optimizer
    beta1, beta2 = 0.9, 0.999
    optimizer = optim.Adam(model.parameters(), lr=p.LEARNING_RATE, betas=(beta1, beta2))

    # --- schedular
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    return optimizer, scheduler

# ----------------------------------------------------------------------------------------------------------------------
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
