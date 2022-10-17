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

    # --- trn/val loop
    for epoch in range(p.NB_EPOCH):
        print('boucle training main')
        trn.trn_loop_seg(p, networks, segLoss, optimizers, schedulers, logger, trn_dataloader, epoch, device)
        val.val_loop_seg(p, networks, segLoss, logger, val_dataloader, epoch, device)
        logger.save_best_model(epoch, networks)

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
def config_wandb():
    """ TODO
    Args:
        TODO
    Returns:
        TODO
    """

    config = {}
    project_name = {}

    return config, project_name

# ----------------------------------------------------------------------------------------------------------------------
def fetch_optimizer(p, model):
    """ Create the optimizer and learning rate scheduler.
    Args:
        TODO
    Returns:
        TODO
    """

    # --- optimizer
    beta1, beta2 = 0.9, 0.999
    optimizer = optim.Adam(model.parameters(), lr=p.LEARNING_RATE, betas=(beta1, beta2))

    # --- schedular
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    return optimizer, scheduler

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
