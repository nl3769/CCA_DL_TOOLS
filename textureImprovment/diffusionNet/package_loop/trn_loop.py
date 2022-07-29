import torch

import numpy                                as np
from tqdm                                   import tqdm
from package_network.forward_pass           import forward_diffusion_sample
from package_visualization.display_res      import display_dr_trn

# -----------------------------------------------------------------------------------------------------------------------
def trn_loop(p, dataloader, model, loss, optimizer, scheduler, epoch, device, beta_scheduler, logger):
    
    # --- set model in training mode
    model.train()
    
    # --- save loss
    store_loss = []
    metrics_trn = {
        'l1': [], 
        'l2': []
        }

    save = True
    for step, (org, sim, fname) in enumerate(tqdm(dataloader, ascii=True, desc=f'TRN - Epoch nb.: {epoch}')):

        # --- load data
        org, sim = org.to(device), sim.to(device)

        for id_t in range(p.TIME_STEP):

            optimizer.zero_grad()
            t = torch.full((p.BATCH_SIZE,), id_t, device=device).long()
            # t = torch.randint(0, p.TIME_STEP, (p.BATCH_SIZE, ), device=device).long()

            x_noisy_sim, noise_sim = forward_diffusion_sample(
                beta_scheduler.sqrt_alphas_cumprod,
                beta_scheduler.sqrt_one_minus_alphas_cumprod,
                sim,
                t,
                device
                )
            x_noisy_org, noise_org = forward_diffusion_sample(
                beta_scheduler.sqrt_alphas_cumprod,
                beta_scheduler.sqrt_one_minus_alphas_cumprod,
                org,
                t,
                device)

            noise_pred = model(x_noisy_sim, t)
            trn_loss = loss(noise_pred, noise_sim)

            # --- backward pass
            trn_loss.backward()
            optimizer.step()
            scheduler.step()

            # --- save loss
            store_loss.append(trn_loss.detach().cpu())

        if save:
            display_dr_trn(p, model, beta_scheduler, "trn", org[0, ], sim[0, ], fname[0], epoch, device)
            save = False

    store_loss = np.mean(np.array(store_loss))
    print(f'Loss: {store_loss}')
