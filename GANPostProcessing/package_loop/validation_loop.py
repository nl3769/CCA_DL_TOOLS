import os.path

import torch

from tqdm                               import tqdm

import numpy                            as np
import package_utils.utils_evaluation   as puue

# ----------------------------------------------------------------------------------------------------------------------
def validation_loop(discriminator, generator, val_loader, epoch, device, loss, logger, psave):

    # --- if cuda is available then all tensors are to gpu
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # --- initilization to store metrics and losses
    loss_generator_val      = []
    loss_generator_org_val  = {'loss_GAN': [], 'loss_pixel': []}
    loss_discriminator_val  = []
    metrics_validation      = {'l1': [], 'l2': []}

    # --- set model in evaluation mode
    discriminator.eval()
    generator.eval()

    # --- save one image during validation loop
    save = True

    # --- VALIDATION LOOP
    for i_batch, (org, sim, fname) in enumerate(tqdm(val_loader, ascii=True, desc=f'VALIDATION - Epoch nb.: {epoch}')):
        # --- load data
        org, sim = org.to(device), sim.to(device)

        # --- create labels
        valid = Tensor(np.ones((org.size(0), 1, 1, 1)), device=device)
        fake = Tensor(np.zeros((org.size(0), 1, 1, 1)), device=device)

        # --- get prediction
        fake_org = generator(sim)
        pred_fake = discriminator(org, fake_org)

        # --- get loss
        loss_generator, loss_gen_org = loss(org, sim, fake_org, [pred_fake], valid, fake, generator=True)

        # --- real loss
        pred_real = discriminator(org, sim)
        pred_fake = discriminator(org, fake_org.detach())
        loss_discriminator = loss(org, sim, fake_org, [pred_real, pred_fake], valid, fake, generator=False)

        loss_generator_val.append(loss_generator.cpu().detach().numpy())
        loss_generator_org_val['loss_GAN'].append(loss_gen_org['loss_GAN'].cpu().detach().numpy())
        loss_generator_org_val['loss_pixel'].append(loss_gen_org['loss_pixel'].cpu().detach().numpy())
        loss_discriminator_val.append(loss_discriminator.cpu().detach().numpy())
        metrics_validation['l1'].append(loss.compute_L1(org, fake_org).cpu().detach().numpy())
        metrics_validation['l2'].append(loss.compute_L2(org, fake_org).cpu().detach().numpy())

        if save == True:

            fake_org, org, sim = fake_org.cpu().detach().numpy(), org.cpu().detach().numpy(), sim.cpu().detach().numpy()
            fake_org, org, sim = fake_org[0, 0, ], org[0, 0, ], sim[0, 0, ]
            path = os.path.join(psave, "validation_epoch_" + str(epoch) + "_" + fname[0] + ".png")
            puue.save_img_res(org, fake_org, sim, path)
            save = False

    loss_generator_val = np.mean(np.array(loss_generator_val))
    loss_discriminator_val = np.mean(np.array(loss_generator_val))
    loss_generator_org_val['loss_GAN'] = np.mean(np.array(loss_generator_org_val['loss_GAN']))
    loss_generator_org_val['loss_pixel'] = np.mean(np.array(loss_generator_org_val['loss_pixel']))
    logger.add_loss(loss_generator_val, loss_discriminator_val, loss_generator_org_val, set='validation')
    logger.display_loss(epoch=epoch)

    metrics_validation['l1'] = np.mean(np.array(metrics_validation['l1']))
    metrics_validation['l2'] = np.mean(np.array(metrics_validation['l2']))

    return discriminator, generator