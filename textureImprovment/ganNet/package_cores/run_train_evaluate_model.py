import os

import argparse
import importlib
import torch
import math
import wandb

from package_network.load_model                 import load_model
from package_utils.model_information            import save_print
from package_utils.weights_init                 import weights_init
from package_dataloader.fetch_data_loader       import fetch_dataloader
from package_loss.loss                          import lossClass
from package_logger.logger                      import loggerClass
from package_loop                               import training_loop, validation_loop
from package_utils.get_param_wandb              import get_param_wandb
from package_cores.run_evaluate_model           import evaluation

# Set wandb in silent mode (remove display in consol)
os.environ["WANDB_SILENT"] = "true"

################################
# --- RUN TRAINING ROUTINE --- #
################################

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device: ', device)

    # --- load model
    discriminator, generator = load_model(p)

    discriminator = discriminator.to(device)
    generator = generator.to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # --- print model in txt file
    save_print(generator, p.PATH_PRINT_MODEL)

    # --- create trn_loader/val_loader
    trn_loader, len_dataset_trn = fetch_dataloader(p, set = 'training',   shuffle = True,  data_aug = True)
    val_loader, len_dataset_val = fetch_dataloader(p, set = 'validation', shuffle = True, data_aug = False)

    # --- Optimizers
    optimizer_generator, optimizer_discriminator, scheduler_gen, scheduler_discr = fetch_optimizer(p, generator, discriminator, n_step=math.ceil(len_dataset_trn / p.BATCH_SIZE) * p.NB_EPOCH)

    # --- Loss class
    loss = lossClass(p)

    # --- logger (store metrics, history...)
    logger = loggerClass(discriminator, generator, p)

    # --- init weight and biases
    config = get_param_wandb(p, {})
    wandb.init(project="realisticUSTextureGAN_V2", entity="nl37", dir=p.PATH_RES, config=config, name=p.PATH_RES.split('/')[-1])

    for epoch in range(p.NB_EPOCH):
        trn_out = training_loop.trn_loop(discriminator, generator, trn_loader, epoch, device, optimizer_generator, optimizer_discriminator, loss, logger, p.PATH_RANDOM_PRED_TRN)
        val_out = validation_loop.val_loop(discriminator, generator, val_loader, epoch, device, loss, logger, p.PATH_RANDOM_PRED_TRN)
        logger.save_best_model(epoch=epoch, model=generator)
        wandb.log({
            'trn_loss_gen': trn_out[0],
            'trn_loss_GAN': trn_out[1],
            'trn_loss_pxl': trn_out[2],
            'trn_l1':       trn_out[3],
            'trn_l2':       trn_out[4],
            'val_loss_gen': val_out[0],
            'val_loss_GAN': val_out[1],
            'val_loss_pxl': val_out[2],
            'val_l1':       val_out[3],
            'val_l2':       val_out[4],
        })

        scheduler_gen.step()
        scheduler_discr.step()
        
        if logger.get_criterium_early_stop():
            break

    logger.plot_loss()
    logger.plot_metrics()
    logger.save_model_history()

    tst_eval = evaluation('testing', p)
    val_eval = evaluation('validation', p)
    trn_eval = evaluation('training', p)

    wandb.log({
        'tst_eval_EPE_mean':    tst_eval[0],
        'tst_eval_EPE_std':     tst_eval[1],
        'tst_eval_PSNR_mean':   tst_eval[2],
        'tst_eval_PSNR_std':    tst_eval[3],
        'val_eval_EPE_mean':    val_eval[0],
        'val_eval_EPE_std':     val_eval[1],
        'val_eval_PSNR_mean':   val_eval[2],
        'val_eval_PSNR_std':    val_eval[3],
        'trn_eval_EPE_mean':    trn_eval[0],
        'trn_eval_EPE_std':     trn_eval[1],
        'trn_eval_PSNR_mean':    trn_eval[2],
        'trn_eval_PSNR_std':     trn_eval[3],
    })


# ----------------------------------------------------------------------------------------------------------------------
def display_loss(validation: bool, epoch: int, loss_training: float, loss_org_train: float,  loss_val: float, loss_org_val: float):
    """ Display loss at the end of each epoch. """

    # --- display loss
    if validation:
        loss_GAN_train = loss_org_train['loss_GAN']
        loss_pixel_train = loss_org_train['loss_pixel']
        loss_GAN_val = loss_org_val['loss_GAN']
        loss_pixel_val = loss_org_val['loss_pixel']

        print(f'EPOCH {epoch} --- training loss: {loss_training}  / loss_GAN: {loss_GAN_train} / loss_pixel: {loss_pixel_train}')
        print(f'EPOCH {epoch} --- validation loss: {loss_val}  / loss_GAN: {loss_GAN_val} / loss_pixel: {loss_pixel_val}')
    else:
        loss_GAN_train = loss_org_train['loss_GAN']
        loss_pixel_train = loss_org_train['loss_pixel']
        print(f'EPOCH {epoch} --- training loss: {loss_training}  / loss_GAN: {loss_GAN_train} / loss_pixel: {loss_pixel_train}')

# ----------------------------------------------------------------------------------------------------------------------
def fetch_optimizer(p, generator, discriminator, n_step):
    """ Create the optimizer and learning rate scheduler. """

    beta1, beta2 = 0.9, 0.999

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=p.LEARNING_RATE, betas=(beta1, beta2))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=p.LEARNING_RATE, betas=(beta1, beta2))

    # --- schedular
    param = \
        {'max_lr'           : p.LEARNING_RATE,
         'total_steps'      : n_step,
         'epochs'           : p.NB_EPOCH,
         'pct_start'        : 0.05,
         'cycle_momentum'   : False,
         'anneal_strategy'  : 'linear'}

    scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_generator, step_size=p.LEARNING_RATE, gamma=0.5)
    scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=p.LEARNING_RATE, gamma=0.5)

    return optimizer_generator, optimizer_discriminator, scheduler_gen, scheduler_disc

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
