import torch
import os

import numpy                                as np
import matplotlib.pyplot                    as plt

from torchvision                            import transforms
from package_network.forward_pass           import get_index_from_list

# -----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_timestep(model, x, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    """ Calls the model to predict the noise in the image and returns the denoised image. Applies noise to this image,
    if we are not in the last step yet. """

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# -----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_plot_image(model, I, time_step, beta_scheduler, psave, device):

    # Sample noise
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(time_step / num_images)
    I = I[None, ...]
    for i in range(0, time_step)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        I = sample_timestep(
            model,
            I,
            t,
            beta_scheduler.betas,
            beta_scheduler.sqrt_one_minus_alphas_cumprod,
            beta_scheduler.sqrt_recip_alphas,
            beta_scheduler.posterior_variance)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i / stepsize + 1)
            show_tensor_image(I.detach().cpu())

    plt.savefig(psave)

    return I

# -----------------------------------------------------------------------------------------------------------------------
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

# ----------------------------------------------------------------------------------------------------------------------
def save_img_final(org: np.ndarray, fake_org: np.ndarray, sim: np.ndarray, pres: str):

    """Save images in pres folders. """

    plt.figure()


    #plt.rcParams['text.usetex'] = True
    # --- IMAGES
    # --- ORG
    plt.subplot(1, 3, 1)
    plt.imshow(org, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.colorbar()

    # --- FAKE_ORG
    plt.subplot(1, 3, 2)
    plt.imshow(fake_org, cmap='gray')
    plt.axis('off')
    plt.title('Diffu. output')
    plt.colorbar()

    # --- sim
    plt.subplot(1, 3, 3)
    plt.imshow(sim, cmap='gray')
    plt.axis('off')
    plt.title('Simulated')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(pres, bbox_inches='tight', dpi=1000)

    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def display_dr_trn(p, model, beta_scheduler, set, org, sim, fname, epoch, device):

    psave_diffusion = os.path.join(p.PATH_RDM_PRED_DIFFUSION, set + "_" + str(epoch) + "_" + fname + ".png")
    psave_res = os.path.join(p.PATH_RDM_PRED_FINAL_RES, set + "_" + str(epoch) + "_" + fname + ".png")

    final_res = sample_plot_image(model, sim, p.TIME_STEP, beta_scheduler, psave_diffusion, device)
    save_img_final(
        org=org.detach().cpu().numpy().squeeze(),
        fake_org=final_res.detach().cpu().numpy().squeeze(),
        sim=sim.detach().cpu().numpy().squeeze(),
        pres=psave_res)