import torch

class diffusionLoss():

    # -------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):

        self.loss = lambda noise, noise_pred: self.compute_L1(noise, noise_pred)

    # -------------------------------------------------------------------------------------------------------------------
    def compute_L1(self, noise, noise_pred):

        L1_metric = torch.nn.L1Loss(reduction='mean')

        return L1_metric(noise, noise_pred)

    # -------------------------------------------------------------------------------------------------------------------
    def __call__(self, noise, noise_pred):
        val = self.loss(noise, noise_pred)
        
        return val
