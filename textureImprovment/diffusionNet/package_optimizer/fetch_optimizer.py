import torch

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def fetch_optimizer(p, model, n_step):
    """ Create the optimizer and learning rate scheduler. """

    beta1, beta2 = 0.9, 0.999

    optimizer = torch.optim.Adam(model.parameters(), lr=p.LEARNING_RATE, betas=(beta1, beta2))

    # --- schedular
    param = {
        'max_lr'           : p.LEARNING_RATE,
        'total_steps'      : n_step,
        'epochs'           : p.NB_EPOCH,
        'pct_start'        : 0.05,
        'cycle_momentum'   : False,
        'anneal_strategy'  : 'linear'
        }

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p.LEARNING_RATE, gamma=0.5)

    return optimizer, scheduler

# ----------------------------------------------------------------------------------------------------------------------------------------------------
