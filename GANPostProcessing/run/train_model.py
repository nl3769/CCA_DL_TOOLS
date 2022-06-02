import os

import argparse
import importlib
import torch
import math

import numpy                                    as np

from tqdm                                       import tqdm
from package_utils.load_model                   import load_model
from package_utils.model_information            import save_print
from package_utils.weights_init                 import weights_init
from package_dataloader.fetch_data_loader       import fetch_dataloader
from package_utils.visualize_data               import visualize_images, save_inputs
from package_loss.loss                          import lossClass
from package_logger.logger                      import loggerClass
import wandb

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
    # discriminator.apply(weights_init)
    # generator.apply(weights_init)

    # --- print model in txt file
    save_print(generator, p.PATH_RES)

    # --- create train_loader/val_loader
    train_loader, len_dataset_training = fetch_dataloader(p, set='training', shuffle=True, data_aug=True)
    val_loader, len_dataset_validation = fetch_dataloader(p, set='validation', shuffle=False, data_aug=False)

    # --- Optimizers
    optimizer_generator, optimizer_discriminator = fetch_optimizer(p, generator, discriminator, n_step=math.ceil(
        len_dataset_training / p.BATCH_SIZE) * p.NB_EPOCH)

    # --- Loss class
    loss = lossClass(p)

    # --- logger (store metrics, history...)
    logger = loggerClass(discriminator, generator, p)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # --- init weight and biases
    config = {"learning_rate": p.LEARNING_RATE,
              "epochs": p.NB_EPOCH,
              "early_stop": p.EARLY_STOP,
              "batch_size": p.BATCH_SIZE,
              "loss": p.LOSS,
              "normalization": p.NORMALIZATION,
              "kernel_size": p.KERNEL_SIZE,
              "cascade_kernel": p.CASCADE_FILTERS,
              "dropout": p.DROPOUT,
              "lambda_pixel": p.LOSS_BALANCE['lambda_pixel'],
              "lambda_GAN": p.LOSS_BALANCE['lambda_GAN'],
              "model_name": p.MODEL_NAME}
    # wandb.init(project="realisticUSTextureGAN", entity="nl37", dir=p.PATH_RES, config=config,
    #            name=p.PATH_RES.split('/')[-1])

    for epoch in range(p.NB_EPOCH):

        # --- set model in training mode
        loss_generator_train = []
        loss_generator_org_train = {'loss_GAN': [], 'loss_pixel': []}
        loss_discriminator_train = []
        metrics_training = {'l1': [], 'l2': []}

        loss_generator_val = []
        loss_generator_org_val = {'loss_GAN': [], 'loss_pixel': []}
        loss_discriminator_val = []
        metrics_validation = {'l1': [], 'l2': []}

        # --- TRAINING LOOP
        discriminator.train()
        generator.train()
        for i_batch, (org, sim, _) in enumerate(tqdm(train_loader, ascii=True, desc=f'TRAINING - Epoch nb.: {epoch}')):
            # --- load data
            org, sim = org.to(device), sim.to(device)

            # --- Create labels
            valid = Tensor(np.ones((org.size(0), 1, 1, 1)), device=device)
            fake = Tensor(np.zeros((org.size(0), 1, 1, 1)), device=device)

            # --- TRAIN GENERATOR
            optimizer_generator.zero_grad()

            # --- GAN loss
            fake_org = generator(sim)
            pred_fake = discriminator(org, fake_org)

            # --- get generator loss
            loss_generator, loss_gen_org = loss(org, sim, fake_org, [pred_fake], valid, fake, generator=True)

            # --- Compute the gradient and perform one optimization step
            loss_generator.backward()
            optimizer_generator.step()

            # --- TRAIN DISCRIMINATOR
            optimizer_discriminator.zero_grad()

            # --- Real loss
            pred_real = discriminator(org, sim)
            fake_org = generator(sim)
            pred_fake = discriminator(org, fake_org.detach())
            loss_discriminator = loss(org, sim, fake_org, [pred_real, pred_fake], valid, fake, generator=False)

            # --- Compute the gradient and perform one optimization step
            loss_discriminator.backward()
            optimizer_discriminator.step()

            loss_generator_train.append(loss_generator.cpu().detach().numpy())
            loss_generator_org_train['loss_GAN'].append(loss_gen_org['loss_GAN'].cpu().detach().numpy())
            loss_generator_org_train['loss_pixel'].append(loss_gen_org['loss_pixel'].cpu().detach().numpy())
            loss_discriminator_train.append(loss_discriminator.cpu().detach().numpy())
            metrics_training['l1'].append(loss.compute_L1(org, fake_org).cpu().detach().numpy())
            metrics_training['l2'].append(loss.compute_L2(org, fake_org).cpu().detach().numpy())

        loss_generator_train = np.mean(np.array(loss_generator_train))
        loss_discriminator_train = np.mean(np.array(loss_discriminator_train))
        loss_generator_org_train['loss_GAN'] = np.mean(np.array(loss_generator_org_train['loss_GAN']))
        loss_generator_org_train['loss_pixel'] = np.mean(np.array(loss_generator_org_train['loss_pixel']))
        logger.add_loss(loss_generator_train, loss_discriminator_train, loss_generator_org_train, set='training')

        # --- VALIDATION LOOP
        discriminator.eval()
        generator.eval()
        for i_batch, (org, sim, _) in enumerate(tqdm(val_loader, ascii=True, desc=f'VALIDATION - Epoch nb.: {epoch}')):
            # --- load data
            org, sim = org.to(device), sim.to(device)

            # --- Create labels
            valid = Tensor(np.ones((org.size(0), 1, 1, 1)), device=device)
            fake = Tensor(np.zeros((org.size(0), 1, 1, 1)), device=device)

            # --- TRAIN GENERATOR
            optimizer_generator.zero_grad()

            # --- GAN loss
            fake_org = generator(sim)
            pred_fake = discriminator(org, fake_org)

            # --- get generator loss
            loss_generator, loss_gen_org = loss(org, sim, fake_org, [pred_fake], valid, fake, generator=True)

            # --- Real loss
            pred_real = discriminator(org, sim)
            fake_org = generator(sim)
            pred_fake = discriminator(org, fake_org.detach())
            loss_discriminator = loss(org, sim, fake_org, [pred_real, pred_fake], valid, fake, generator=False)

            loss_generator_val.append(loss_generator.cpu().detach().numpy())
            loss_generator_org_val['loss_GAN'].append(loss_gen_org['loss_GAN'].cpu().detach().numpy())
            loss_generator_org_val['loss_pixel'].append(loss_gen_org['loss_pixel'].cpu().detach().numpy())
            loss_discriminator_val.append(loss_discriminator.cpu().detach().numpy())
            metrics_validation['l1'].append(loss.compute_L1(org, fake_org).cpu().detach().numpy())
            metrics_validation['l2'].append(loss.compute_L2(org, fake_org).cpu().detach().numpy())

        loss_generator_val = np.mean(np.array(loss_generator_val))
        loss_discriminator_val = np.mean(np.array(loss_generator_val))
        loss_generator_org_val['loss_GAN'] = np.mean(np.array(loss_generator_org_val['loss_GAN']))
        loss_generator_org_val['loss_pixel'] = np.mean(np.array(loss_generator_org_val['loss_pixel']))
        logger.add_loss(loss_generator_val, loss_discriminator_val, loss_generator_org_val, set='validation')
        logger.display_loss(epoch=epoch)
        logger.save_best_model(epoch=epoch, model=generator)

        metrics_training['l1'] = np.mean(np.array(metrics_training['l1']))
        metrics_training['l2'] = np.mean(np.array(metrics_training['l2']))
        metrics_validation['l1'] = np.mean(np.array(metrics_validation['l1']))
        metrics_validation['l2'] = np.mean(np.array(metrics_validation['l2']))

        wandb.log({"loss_generator_train": loss_generator_train,
                   "loss_discriminator_train": loss_discriminator_train,
                   "loss_generator_val": loss_generator_val,
                   "loss_discriminator_val": loss_discriminator_val,
                   "l1_training": metrics_training['l1'],
                   "l2_training": metrics_training['l2'],
                   "l1_val": metrics_validation['l1'],
                   "l2_val": metrics_validation['l2']})

        if logger.get_criterium_early_stop():
            break

    logger.plot_loss()
    # logger.plot_metrics()
    logger.save_model_history()

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
    param = {'max_lr': p.LEARNING_RATE,
             'total_steps': n_step,
             'epochs': p.NB_EPOCH,
             'pct_start': 0.05,
             'cycle_momentum': False,
             'anneal_strategy': 'linear'}

    return optimizer_generator, optimizer_discriminator

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------