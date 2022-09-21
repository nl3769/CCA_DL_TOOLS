import os

import argparse
import importlib

import matplotlib.pyplot as plt
import torch

import numpy                                    as np

from PIL                                        import Image
from tqdm                                       import tqdm
from package_network.load_model                 import load_model
from package_dataloader.fetch_data_loader       import fetch_dataloader

################################
# --- RUN TRAINING ROUTINE --- #
################################

def inference(generator, data_loader, p, device):

    generator.eval()

    # --- evaluation loop
    for i_batch, (org, sim, fname) in enumerate(tqdm(data_loader, ascii=True, desc='SAVE_PRED')):

        # --- load data
        org, sim = org.to(device), sim.to(device)
        name = fname[0]

        # --- get prediction
        fake_org = generator(sim)

        # --- save image
        org = Image.fromarray(np.array(org.detach().cpu()).squeeze()).convert("L")
        fake_org = Image.fromarray(np.array(fake_org.detach().cpu()).squeeze()).convert("L")
        sim = Image.fromarray(np.array(sim.detach().cpu()).squeeze()).convert("L")

        org.save(os.path.join(p.PATH_RES_ORG, name + ".png"), format="png")
        fake_org.save(os.path.join(p.PATH_RES_GAN, name + ".png"), format="png")
        sim.save(os.path.join(p.PATH_RES_SIM, name + ".png"), format="png")

        if p.NB_SAVE <= i_batch:
            break

# ----------------------------------------------------------------------------------------------------------------------
def main():

    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of parameters required to execute the code.')

    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device: ', device)

    # --- load model
    _, generator = load_model(p)
    generator = generator.to(device)

    # --- create trn_loader/val_loader
    trn_loader, len_dataset_trn = fetch_dataloader(p, set = 'training',   shuffle = True, data_aug = False)
    val_loader, len_dataset_val = fetch_dataloader(p, set = 'validation', shuffle = True, data_aug = False)
    tst_loader, len_dataset_tst = fetch_dataloader(p, set ='testing',     shuffle = True, data_aug = False)

    # for id_img in range(p.NB_SAVE):
    inference(generator, trn_loader, p, device)
    inference(generator, val_loader, p, device)
    inference(generator, tst_loader, p, device)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
