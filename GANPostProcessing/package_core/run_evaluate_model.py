import argparse
import importlib
import os
import torch
import math

import pandas                                       as pd
import numpy                                        as np

from tqdm                                           import tqdm
from package_utils.load_model                       import load_model
from package_dataloader.fetch_data_loader           import fetch_dataloader
from package_loss.loss                              import lossClass
from package_utils.utils_evaluation                 import get_path_model, save_evaluation_res, save_pred


# ----------------------------------------------------------------------------------------------------------------------
def main(set):
    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    # --- device configuration
    print(p)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device: ', device)

    # --- create path to store prediction
    pres_img = os.path.join(p.PATH_RES, 'images_' + set)
    if not os.path.exists(pres_img):
        os.makedirs(pres_img)

    # --- load model
    _, model = load_model(p)
    path = get_path_model(p.PATH_RES)
    id = 1 if len(path) == 2 else 0  # we use model based on validation if it exists
    id = 0
    model.load_state_dict(torch.load(path[id]))
    model = model.to(device)
    model.eval()

    # --- load dataset
    test_loader, test_dataset = fetch_dataloader(p, set=set, shuffle=True, evaluation=True)

    # --- loss
    loss = lossClass(p)

    # --- set model in evaluation mode (it is not the same behaviour, it removes the dropouts...)
    model.eval()

    # --- variable to store metrics
    test_loss = []
    test_metrics = {}
    keys = ['EPE_org_vs_sim', 'EPE_org_vs_fakeOrg', 'PSNR_org_vs_sim', 'PSNR_org_vs_fakeOrg', 'name']
    for key in keys:
        test_metrics[key] = []

    # --- evaluation loop
    for i_batch, (org, sim, fname) in enumerate(tqdm(test_loader, ascii=True, desc='EVALUATION')):
        # --- load data
        org, sim = org.to(device), sim.to(device)
        name = fname[0]

        # --- get prediction
        fake_org = model(sim)

        # --- compute metrics
        metric_org, metric_pred = compute_EPE(org, sim, fake_org)
        PSNR_org_sim, PSNR_org_fakeOrg = compute_PSNR(org, sim, fake_org)
        # test_loss.append(loss_test.cpu().detach().numpy())

        test_metrics['EPE_org_vs_sim'].append(metric_org)
        test_metrics['EPE_org_vs_fakeOrg'].append(metric_pred)
        test_metrics['PSNR_org_vs_sim'].append(PSNR_org_sim)
        test_metrics['PSNR_org_vs_fakeOrg'].append(PSNR_org_fakeOrg)
        test_metrics['name'].append(name)
        save_pred(org, fake_org, sim, pres_img, name)

    df = pd.DataFrame(data=test_metrics)
    df.to_csv(os.path.join(p.PATH_RES, 'res_evaluation_' + set + '.csv'))
    #    save_evaluation_res(test_metrics, p.PATH_RES, loss.metrics)
    save_metrics(p.PATH_RES, set, test_metrics)
    # save_prediction_fig(p.PATH_RES, gt, names_img, pred_it, inputs)
    # save_prediction_it_fig(p.PATH_RES, gt, names_img, pred_it, inputs)

# ----------------------------------------------------------------------------------------------------------------------
def save_metrics(pres, set, metric):


    f = open(os.path.join(pres, 'metrics_' + set + '.txt'), 'w')
    for key in metric.keys():
        if key != 'name':
            values = np.array(metric[key])
            mean = np.mean(values)
            std = np.std(values)

            f.write(key + 'n')
            f.write('mean: ' + str(mean) + '\n')
            f.write('std: ' + str(std) + '\n')
            f.write('\n' + '###############' +'\n')
    f.close()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main('test')

# ----------------------------------------------------------------------------------------------------------------------
