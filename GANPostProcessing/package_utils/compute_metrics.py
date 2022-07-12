import numpy                        as np

# ----------------------------------------------------------------------------------------------------------------------

def compute_EPE(org, sim, fake_org):

    org = org.cpu().detach().numpy()
    sim = sim.cpu().detach().numpy()
    fake_org = fake_org.cpu().detach().numpy()

    EPE_org = np.mean(np.abs(org - sim))
    EPE_fake = np.mean(np.abs(org - fake_org))

    return EPE_org, EPE_fake
# ----------------------------------------------------------------------------------------------------------------------