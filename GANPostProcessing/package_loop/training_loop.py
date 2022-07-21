import torch
import os

from tqdm                               import tqdm
import numpy                            as np

import package_utils.utils_evaluation   as puue

# ----------------------------------------------------------------------------------------------------------------------
def trn_loop(discriminator, generator, trn_loader, epoch, device, optimizer_generator, optimizer_discriminator, loss, logger, psave):

    # --- if cuda is available then all tensors are on gpu
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # --- initilization to store metrics and losses
    loss_generator_trn      = []
    loss_generator_org_trn  = {'loss_GAN': [], 'loss_pixel': []}
    loss_discriminator_trn  = []
    metrics_trn             = {'l1': [], 'l2': []}

    # --- set model in trn mode
    discriminator.train()
    generator.train()

    # --- trn LOOP
    save = True
    for i_batch, (org, sim, fname) in enumerate(tqdm(trn_loader, ascii=True, desc=f'TRN - Epoch nb.: {epoch}')):

        # --- load data
        org, sim = org.to(device), sim.to(device)

        # --- Create labels
        valid   = Tensor(np.ones((org.size(0), 1, 1, 1))).to(device)
        fake    = Tensor(np.zeros((org.size(0), 1, 1, 1))).to(device)

        # --- trn generator
        optimizer_generator.zero_grad()

        # --- Get predictions
        fake_org = generator(sim)
        pred_fake = discriminator(sim, fake_org)

        # --- Get generator loss
        loss_generator, loss_gen_org = loss(org, sim, fake_org, [pred_fake], valid, fake, generator=True)

        # --- Compute the gradient and perform one optimization step
        loss_generator.backward()
        optimizer_generator.step()

        # --- trn DISCRIMINATOR
        optimizer_discriminator.zero_grad()

        # --- Real loss
        pred_real = discriminator(org, sim)
        fake_org = generator(sim)
        pred_fake = discriminator(org, fake_org.detach())
        loss_discriminator = loss(org, sim, fake_org, [pred_real, pred_fake], valid, fake, generator=False)

        # --- Compute the gradient and perform optimization step
        loss_discriminator.backward()
        optimizer_discriminator.step()

        loss_generator_trn.append(loss_generator.cpu().detach().numpy())
        loss_generator_org_trn['loss_GAN'].append(loss_gen_org['loss_GAN'].cpu().detach().numpy())
        loss_generator_org_trn['loss_pixel'].append(loss_gen_org['loss_pixel'].cpu().detach().numpy())
        loss_discriminator_trn.append(loss_discriminator.cpu().detach().numpy())
        metrics_trn['l1'].append(loss.compute_L1(org, fake_org).cpu().detach().numpy())
        metrics_trn['l2'].append(loss.compute_L2(org, fake_org).cpu().detach().numpy())

        if save == True:
            fake_org, org, sim = fake_org.cpu().detach().numpy(), org.cpu().detach().numpy(), sim.cpu().detach().numpy()
            fake_org, org, sim = fake_org[0, 0, ], org[0, 0, ], sim[0, 0, ]
            path = os.path.join(psave, "trn_epoch_" + str(epoch) + "_" + fname[0] + ".png")
            puue.save_img_res(org, fake_org, sim, path)
            save = False

    loss_generator_trn = np.mean(np.array(loss_generator_trn))
    loss_discriminator_trn = np.mean(np.array(loss_discriminator_trn))
    loss_generator_org_trn['loss_GAN'] = np.mean(np.array(loss_generator_org_trn['loss_GAN']))
    loss_generator_org_trn['loss_pixel'] = np.mean(np.array(loss_generator_org_trn['loss_pixel']))

    logger.add_loss(loss_generator_trn, loss_discriminator_trn, loss_generator_org_trn, set='training')
    metrics_trn['l1'] = np.mean(np.array(metrics_trn['l1']))
    metrics_trn['l2'] = np.mean(np.array(metrics_trn['l2']))
    logger.add_metrics(metrics_trn, 'training')

    return \
        loss_generator_trn, \
        loss_generator_org_trn['loss_GAN'], \
        loss_generator_org_trn['loss_pixel'], \
        metrics_trn['l1'], \
        metrics_trn['l2']
# ----------------------------------------------------------------------------------------------------------------------
