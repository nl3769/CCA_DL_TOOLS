'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''
import os

import torch

import numpy                                as np
import package_utils.pytorch_processing     as pupp

from medpy.metric.binary                    import dc, hd
from tqdm                                   import tqdm

# ----------------------------------------------------------------------------------------------------------------------
def tst_loop_flow(p, networks, segLoss, loader, device, set):

    seg_loss = []
    hausdorff = []
    dice = []

    # --- set model in training mode
    networks["netSeg"].eval()

    for i_batch, (I1, M1, CF, fname) in enumerate(tqdm(loader, ascii=True, desc=f'TST')):

        # --- load data
        I1, M1 = I1.to(device), M1.to(device)

        # --- inference
        M1_pred = networks["netSeg"](I1)

        # --- loss/metrics
        loss, seg_metrics = segLoss(M1_pred, M1)
        seg_loss.append(loss.cpu().detach().numpy())

        for id_batch in range(M1.shape[0]):
            pred = M1_pred[id_batch, ].cpu().detach().numpy().squeeze()
            gt = M1[id_batch, ].cpu().detach().numpy().squeeze()
            x_cf = CF['xCF'][id_batch].cpu().detach().numpy().squeeze()
            y_cf = CF['yCF'][id_batch].cpu().detach().numpy().squeeze()
            hausdorff.append(hd(pred, gt, voxelspacing=(x_cf, y_cf)))
            dice.append(dc(pred, gt))


    seg_loss = np.array(seg_loss)
    hausdorff = np.array(hausdorff)
    dice = np.array(dice)

    with open(os.path.join(p.PATH_TST, set + '_metrics.txt'), 'w') as f:
        f.write("# --- HAUSDORFF DISTANCE \n")
        f.write("mean="+str(hausdorff.mean())+" / std="+str(hausdorff.std()) + " m \n")
        f.write("# --- DICE \n")
        f.write("mean=" + str(dice.mean()) + " / std=" + str(dice.std()) + "\n")
        f.write("# --- LOSS (LOSS DICE + BCE) \n")
        f.write("mean=" + str(seg_loss.mean()) + " / std=" + str(seg_loss.std()))


