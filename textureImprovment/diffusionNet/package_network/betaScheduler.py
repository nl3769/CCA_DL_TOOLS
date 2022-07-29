import torch
import torch.nn.functional                      as F
# ----------------------------------------------------------------------------------------------------------------------------------------------------
def linear_beta_schedule(time_steps, start=0.0001, end=0.02):
    " Create timeline. "

    return torch.linspace(start, end, time_steps)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class betaScheduler():

    def __init__(self, p):

        self.time_step = p.TIME_STEP
        self.betas = linear_beta_schedule(self.time_step, p.BETA[0], p.BETA[1])
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)