import argparse
import importlib
import torch
import math

from package_network.load_model             import load_model
from package_dataloader.fetch_data_loader   import fetch_data_loader
from package_loop.trn_loop                  import trn_loop
from package_loss.diffusionLoss             import diffusionLoss
from package_network.betaScheduler          import betaScheduler
from package_optimizer.fetch_optimizer      import fetch_optimizer
from package_logger.logger                  import loggerClass

# -----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- load model
    model = load_model(p)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- dataloader
    trn_dataloader, len_dataset_trn = fetch_data_loader(p, set= 'training',   shuffle = True, data_aug = False)
    vld_dataloader, _               = fetch_data_loader(p, set= 'validation', shuffle = True, data_aug = False)
    tst_dataloader, _               = fetch_data_loader(p, set= 'testing',    shuffle = True, data_aug = False)

    # --- loss
    loss = diffusionLoss(p)

    # --- Pre-calculate different terms for closed form
    beta_scheduler = betaScheduler(p)

    # --- optimizer
    optimizer, scheduler = fetch_optimizer(p, model, n_step=math.ceil(len_dataset_trn / p.BATCH_SIZE) * p.NB_EPOCH)
    
    # --- logger
    logger = loggerClass(model, scheduler, p)

    for epoch in range(p.NB_EPOCH):
        trn_loop(p, trn_dataloader, model, loss, optimizer, scheduler, epoch, device, beta_scheduler, logger)

# -----------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    main()
