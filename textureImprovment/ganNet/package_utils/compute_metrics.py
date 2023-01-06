'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import numpy                        as np
import math

# ----------------------------------------------------------------------------------------------------------------------
def compute_EPE(org, sim, fake_org):

    org = org.cpu().detach().numpy()
    sim = sim.cpu().detach().numpy()
    fake_org = fake_org.cpu().detach().numpy()

    EPE_org = np.mean(np.abs(org - sim))
    EPE_fake = np.mean(np.abs(org - fake_org))

    return EPE_org, EPE_fake

# ----------------------------------------------------------------------------------------------------------------------
def compute_PSNR(org, sim, fake_org, I_max):

    org = org.cpu().detach().numpy()
    sim = sim.cpu().detach().numpy()
    fake_org = fake_org.cpu().detach().numpy()

    MSE_org_sim = np.mean(np.power(np.abs(org - sim), 2))
    MSE_org_fakeOrg = np.mean(np.power(np.abs(org - fake_org), 2))

    PSNR_org_sim = 10*math.log10(I_max ** 2 / MSE_org_sim)
    PSNR_org_fakeOrg = 10*math.log10(I_max ** 2/ MSE_org_fakeOrg)

    return PSNR_org_sim, PSNR_org_fakeOrg

# ----------------------------------------------------------------------------------------------------------------------
