'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import glob
import torch
import os
import matplotlib
matplotlib.use('Agg')

import numpy                        as np
import matplotlib.pyplot            as plt

# ----------------------------------------------------------------------------------------------------------------------
def save_pred(org: torch.Tensor, fake_org: torch.Tensor, sim: torch.Tensor, pres: str, name: str, set = ''):
    """ Adapt tensor to numpy to save results. """

    org = org.cpu().detach().numpy().squeeze()
    fake_org = fake_org.cpu().detach().numpy().squeeze()
    sim = sim.cpu().detach().numpy().squeeze()

    save_img_res(org, fake_org, sim, os.path.join(pres, set + '_' + name))

# ----------------------------------------------------------------------------------------------------------------------
def save_evaluation_res(res, path: str, keys):
    # --- save result using subplot
    f = plt.figure()
    color = ['r', 'g', 'c', 'b']
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    for id, key in enumerate(keys.keys()):
        plt.subplot(2, 2, id + 1)
        plt.plot(res[key], color=color[id])
        plt.title(key)
    plt.tight_layout()
    f.savefig(os.path.join(path, 'evaluation_res_subplot.png'), dpi=200)
    plt.close()

    # --- superimpose results on the same graph
    f = plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    legend = []
    for id, key in enumerate(keys):
        plt.plot(res[key], color=color[id])
        legend.append(key)
    plt.legend(legend)
    f.savefig(os.path.join(path, 'evaluation_res_full.png'), dpi=200)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def get_path_model(path: str):
    ''' Get model's path. '''

    weights = sorted(glob.glob(os.path.join(path, '*.pth')))
    root = [os.path.join(path, mname) for mname in weights]

    return root

# ----------------------------------------------------------------------------------------------------------------------
def save_img_res(org: np.ndarray, fake_org: np.ndarray, sim: np.ndarray, pres: str):

    """Save images in pres folders. """

    plt.figure()
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
    plt.title('GAN output')
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