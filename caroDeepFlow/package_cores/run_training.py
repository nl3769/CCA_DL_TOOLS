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

import torch.optim                              as optim
import package_dataloader.utils                 as pdlu
import package_network.utils                    as pnu
import package_loss.lossSeg                     as plls
import package_loss.lossFlow                    as plll
import package_logger.logger                    as plog
import package_loop.training_loop               as prtl
import package_loop.validation_loop             as prvl
import package_debug.visualisation              as pdv

from icecream                                   import ic
from package_utils.IMCExtractor                 import IMCExtractor

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

    # --- load models
    netEncoder, netSeg, netFlow = pnu.load_model(p)
    netEncoder  = netEncoder.to(device)
    netSeg      = netSeg.to(device)
    netFlow     = netFlow.to(device)

    # --- get dataloader
    training_dataloader, validation_dataloader, _ = pdlu.fetch_dataloader(p)

    # --- print model in txt file
    pnu.save_print(netEncoder, p.PATH_PRINT_MODEL, 'netEncoder')
    pnu.save_print(netSeg, p.PATH_PRINT_MODEL, 'netSeg')
    pnu.save_print(netFlow, p.PATH_PRINT_MODEL, 'netFlow')

    # --- optimizer/scheduler
    optimizerEncoder, schedulerEncoder = fetch_optimizer(p, netEncoder, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)
    optimizerDecoder, schedulerDecoder = fetch_optimizer(p, netSeg, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)
    optimizerFlow, schedulerFlow = fetch_optimizer(p, netFlow, n_step=math.ceil(len(training_dataloader.dataset.image_list) / p.BATCH_SIZE) * p.NB_EPOCH)

    # --- losses
    segLoss = plls.lossSeg()
    flowLoss = plll.lossFlow()

    # --- store optimizers/schedulers and optimizers
    networks = {
        "netEncoder": netEncoder,
        "netSeg": netSeg,
        "netFlow": netFlow}
    optimizers = {
        "netEncoder": optimizerEncoder,
        "netSeg": optimizerDecoder,
        "netFlow": optimizerFlow}
    schedulers = {
        "netEncoder": schedulerEncoder,
        "netSeg": schedulerDecoder,
        "netFlow": schedulerFlow}

    # --- logger
    logger = plog.loggerClass(p, segLoss.metrics.keys(), flowLoss.metrics.keys())

    # --- IMC extractor
    IMCExtract = IMCExtractor(p.ADVENTICIA_DIM)

    # --- loop
    for epoch in range(p.NB_EPOCH):

        if p.FEATURES == 'shared':
            prtl.training_loop_shared(p, networks, segLoss, flowLoss, optimizers, schedulers, logger, training_dataloader, epoch, device, IMCExtract)
            prvl.validation_loop_shared(p, networks, segLoss, flowLoss, logger, validation_dataloader, epoch, device, IMCExtract)
        if p.FEATURES == 'split':
            prtl.training_loop_split(p, networks, segLoss, flowLoss, optimizers, schedulers, logger, training_dataloader, epoch, device, IMCExtract)
            prvl.validation_loop_split(p, networks, segLoss, flowLoss, logger, validation_dataloader, epoch, device, IMCExtract)

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
    param = {
        'max_lr': p.LEARNING_RATE,
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
