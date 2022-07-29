import flow_vis
import numpy as np
import glob
import torch
import os

import matplotlib.pyplot            as plt
import package_utils.utils          as puu


# ----------------------------------------------------------------------------------------------------------------------
def save_pred(org: torch.Tensor, fake_org: torch.Tensor, sim: torch.Tensor, pres: str, name: str, set = ''):
    """ Adapt tensor to numpy to save results. """

    org = org.cpu().detach().numpy().squeeze()
    fake_org = fake_org.cpu().detach().numpy().squeeze()
    sim = sim.cpu().detach().numpy().squeeze()

    save_img_res(org, fake_org, sim, os.path.join(pres, set + '_' + name))


# ----------------------------------------------------------------------------------------------------------------------
def save_prediction_it_fig(res, gt, fname, pred, inputs):
    rpath = os.path.join(res, 'images')
    puu.check_dir(rpath)

    # save figure (img1, img2, GT, final flow)
    for pid in range(len(fname)):
        # --- get img1 and img2
        img1, img2 = inputs[pid][0], inputs[pid][1]
        img1, img2 = img1.numpy(), img2.numpy()
        img1, img2 = img1[0, ...], img2[0, ...]

        # --- get gt
        gt_ = gt[pid]
        gt_ = gt_.numpy()
        gt_ = gt_[0, ...]
        gt_ = np.transpose(gt_, axes=[1, 2, 0])
        gt_ = flow_vis.flow_to_color(gt_, convert_to_bgr=False)
        prediction_ = []
        for it in range(len(pred[0])):
            # --- get prediction
            pred_ = pred[pid][it].detach()
            pred_ = pred_.numpy()
            pred_ = pred_[0, ...]
            pred_ = np.transpose(pred_, axes=[1, 2, 0])
            pred_ = flow_vis.flow_to_color(pred_, convert_to_bgr=False)

            prediction_.append(pred_)

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)

        for id in range(len(prediction_)):
            plt.subplot(3, 4, id + 1)
            plt.imshow(prediction_[id].astype(int))
            plt.title(f'it. {id + 1}')

        plt.tight_layout()
        plt.savefig(os.path.join(rpath, fname[pid] + '_iteration.svg'), dpi=200)

        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def save_prediction_fig(res: str, gt: np.ndarray, name: str, predictions: torch.Tensor, inputs: torch.Tensor):
    rpath = os.path.join(res, 'images')
    puu.check_dir(rpath)

    # save figure (img1, img2, GT, final flow)
    for pid in range(len(name)):
        # --- get img1 and img2
        img1, img2 = inputs[pid][0], inputs[pid][1]
        img1, img2 = img1.numpy(), img2.numpy()
        img1, img2 = img1[0, ...], img2[0, ...]

        # --- get gt
        gt_ = gt[pid]
        gt_ = gt_.numpy()
        gt_ = gt_[0, ...]
        gt_ = np.transpose(gt_, axes=[1, 2, 0])
        gt_ = flow_vis.flow_to_color(gt_, convert_to_bgr=False)

        # --- get prediction
        prediction = predictions[pid][-1].detach()
        prediction = prediction.numpy()
        prediction = prediction[0, ...]
        prediction = np.transpose(prediction, axes=[1, 2, 0])
        prediction = flow_vis.flow_to_color(prediction, convert_to_bgr=False)

        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)

        plt.subplot(2, 2, 1)
        plt.imshow(img1.transpose(1, 2, 0).astype(int))
        plt.title('img1')

        plt.subplot(2, 2, 2)
        plt.imshow(img2.transpose(1, 2, 0).astype(int))
        plt.title('img2')

        plt.subplot(2, 2, 3)
        plt.imshow(gt_.astype(int))
        plt.title('GT')

        plt.subplot(2, 2, 4)
        plt.imshow(prediction.astype(int))
        plt.title('Pred.')

        plt.tight_layout()
        plt.savefig(os.path.join(rpath, name[pid] + 'pred_vs_GT.svg'), dpi=200)

        plt.close()


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
def display_loss(validation: bool, epoch: int, loss_training: float, loss_validation: float):
    """ Display loss at the end of each epoch. """

    # --- display loss
    if validation:
        print(f'EPOCH {epoch} --- training loss: {loss_training} | validation loss: {loss_validation}')
    else:
        print(f'EPOCH {epoch} --- training loss: {loss_training}')


# ----------------------------------------------------------------------------------------------------------------------
def save_img_res(org: np.ndarray, fake_org: np.ndarray, sim: np.ndarray, pres: str):

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