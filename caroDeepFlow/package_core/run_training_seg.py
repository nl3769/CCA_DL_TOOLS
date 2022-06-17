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
import package_loss.lossFlow 			          as plll
import package_logger.logger 			          as plog
import package_loop.training_loop                 as prtl
import package_loop.validation_loop               as prvl
import package_debug.visualisation 		          as pdv

from icecream 				                      import ic
from package_utils.IMCExtractor                   import IMCExtractor

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
    if p.DEVICE == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    # --- get dataloader
    training_dataloader, validation_dataloader, _ = pdlu.fetch_dataloader_seg(p)

    # --- load models
    netSeg = pnu.load_model_seg(p)
    netSeg = netSeg.to(device)

    # --- optimizer/scheduler
    optimizerDecoder, schedulerDecoder = fetch_optimizer(p, netSeg, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)

    # --- loss
    segLoss = plls.lossSeg()

    # --- store optimizers/schedulers and optimizers
    networks = {"netSeg": netSeg}
    optimizers = {"netSeg": optimizerDecoder}
    schedulers = {"netSeg": schedulerDecoder}

    # --- logger
    logger = plog.loggerClassSeg(p, segLoss.metrics.keys())

    # --- loop
    for epoch in range(p.NB_EPOCH):

        prtl.training_loop_seg(p, networks, segLoss, optimizers, schedulers, logger, training_dataloader, epoch, device)
        prvl.validation_loop_seg(p, networks, segLoss, logger, validation_dataloader, epoch, device)

        logger.save_best_model(epoch, networks)

    logger.plot_loss()
    logger.plot_metrics()
    logger.save_model_history()

# ----------------------------------------------------------------------------------------------------------------------
def config_wandb():
    """ TODO """

    config = {}
    project_name = {}

    return config, project_name

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
