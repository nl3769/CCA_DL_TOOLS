import torch

import torch.nn.functional          as F
import matplotlib.pyplot            as plt
import numpy                        as np

# -----------------------------------------------------------------------------------------------------------------------
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    " Create timeline. "

    return torch.linspace(start, end, timesteps)

# -----------------------------------------------------------------------------------------------------------------------
def get_index_from_list(vals, t, x_shape):
    """ Returns a specific index t of a passed list of values vals while considering the batch dimension. """
    
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# -----------------------------------------------------------------------------------------------------------------------
def forward_diffusion_sample(sqrt_alphas_cumprod,  sqrt_one_minus_alphas_cumprod, x_0, t, device="cpu"):
    """ Takes an image and a timestep as input and returns the noisy version of it. """
    
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# -----------------------------------------------------------------------------------------------------------------------
def forward_pass_toy(trn_dataloader):
    
    # --- beta schedule
    T = 300
    betas = linear_beta_schedule(timesteps=T)
    
    # --- Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    image = next(iter(trn_dataloader))[0]
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    
    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, (idx/stepsize) + 1)
        image, noise = forward_diffusion_sample(sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, image, t)
        show_tensor_image(image)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------------
def forward_pass(image, t):
    
    # --- beta schedule
    T = 300
    betas = linear_beta_schedule(timesteps=T)
    
    # --- Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    image, noise = forward_diffusion_sample(sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, image, t)
    return image, noise
