import torch
import os

from tqdm                               import tqdm
import numpy                            as np

import package_utils.utils_evaluation   as puue

# ----------------------------------------------------------------------------------------------------------------------
def training_loop(discriminator, generator, train_loader, epoch, device, optimizer_generator, optimizer_discriminator, loss, logger, psave):

    # --- if cuda is available then all tensors are to gpu
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # --- initilization to store metrics and losses
    loss_generator_train        = []
    loss_generator_org_train    = {'loss_GAN': [], 'loss_pixel': []}
    loss_discriminator_train    = []
    metrics_training            = {'l1': [], 'l2': []}

    # --- set model in training mode
    discriminator.train()
    generator.train()

    # --- TRAINING LOOP
    save = True
    for i_batch, (org, sim, fname) in enumerate(tqdm(train_loader, ascii=True, desc=f'TRAINING - Epoch nb.: {epoch}')):
        # --- load data
        org, sim = org.to(device), sim.to(device)

        # --- create labels
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

        # --- Compute the gradient and perform optimization step
        loss_discriminator.backward()
        optimizer_discriminator.step()

        loss_generator_train.append(loss_generator.cpu().detach().numpy())
        loss_generator_org_train['loss_GAN'].append(loss_gen_org['loss_GAN'].cpu().detach().numpy())
        loss_generator_org_train['loss_pixel'].append(loss_gen_org['loss_pixel'].cpu().detach().numpy())
        loss_discriminator_train.append(loss_discriminator.cpu().detach().numpy())
        metrics_training['l1'].append(loss.compute_L1(org, fake_org).cpu().detach().numpy())
        metrics_training['l2'].append(loss.compute_L2(org, fake_org).cpu().detach().numpy())

        if save == True:

            fake_org, org, sim = fake_org.cpu().detach().numpy(), org.cpu().detach().numpy(), sim.cpu().detach().numpy()
            fake_org, org, sim = fake_org[0, 0, ], org[0, 0, ], sim[0, 0, ]
            path = os.path.join(psave, "training_epoch_" + str(epoch) + "_" + fname[0] + ".png")
            puue.save_img_res(org, fake_org, sim, path)
            save = False

    loss_generator_train = np.mean(np.array(loss_generator_train))
    loss_discriminator_train = np.mean(np.array(loss_discriminator_train))
    loss_generator_org_train['loss_GAN'] = np.mean(np.array(loss_generator_org_train['loss_GAN']))
    loss_generator_org_train['loss_pixel'] = np.mean(np.array(loss_generator_org_train['loss_pixel']))

    logger.add_loss(loss_generator_train, loss_discriminator_train, loss_generator_org_train, set='training')
    metrics_training['l1'] = np.mean(np.array(metrics_training['l1']))
    metrics_training['l2'] = np.mean(np.array(metrics_training['l2']))

    return discriminator, generator
# ----------------------------------------------------------------------------------------------------------------------