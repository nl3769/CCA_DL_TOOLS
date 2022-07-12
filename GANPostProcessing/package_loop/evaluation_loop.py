import os

from tqdm                                   import tqdm
from package_utils.utils                    import check_dir

import package_utils.compute_metrics        as pucm
import pandas                               as pd
import numpy                                as np

def evaluation_loop(device, data_loader, model, p, set):

    model.eval()

    # --- variable to store metrics
    test_metrics = {}
    keys = ['name', 'EPE_org_vs_sim', 'EPE_org_vs_fakeOrg', 'PSNR_org_vs_sim', 'PSNR_org_vs_fakeOrg']
    for key in keys:
        test_metrics[key] = []

    # --- evaluation loop
    for i_batch, (org, sim, fname) in enumerate(tqdm(data_loader, ascii=True, desc='EVALUATION (' + set + ')')):
        # --- load data
        org, sim = org.to(device), sim.to(device)
        name = fname[0]

        # --- get prediction
        fake_org = model(sim)

        # --- compute metrics
        metric_org, metric_pred = pucm.compute_EPE(org, sim, fake_org)
        PSNR_org_sim, PSNR_org_fakeOrg = pucm.compute_PSNR(org, sim, fake_org)

        # add metrics
        test_metrics['name'].append(name)
        test_metrics['EPE_org_vs_sim'].append(metric_org)
        test_metrics['EPE_org_vs_fakeOrg'].append(metric_pred)
        test_metrics['PSNR_org_vs_sim'].append(PSNR_org_sim)
        test_metrics['PSNR_org_vs_fakeOrg'].append(PSNR_org_fakeOrg)

        # save_pred(org, fake_org, sim, pres_img, name)
        # save_pred(org, fake_org, sim, pres_img, name)

    # --- copute avergae metrics and standart deviation
    EPE_org_vs_sim_mean     = np.array(test_metrics['EPE_org_vs_sim']).mean()
    EPE_org_vs_sim_std      = np.array(test_metrics['EPE_org_vs_sim']).std()
    PSNR_org_vs_sim_mean    = np.array(test_metrics['PSNR_org_vs_sim']).mean()
    PSNR_org_vs_sim_std     = np.array(test_metrics['PSNR_org_vs_sim']).std()

    EPE_org_vs_fakeOrg_mean     = np.array(test_metrics['EPE_org_vs_fakeOrg']).mean()
    EPE_org_vs_fakeOrg_std      = np.array(test_metrics['EPE_org_vs_fakeOrg']).std()
    PSNR_org_vs_fakeOrg_mean    = np.array(test_metrics['PSNR_org_vs_fakeOrg']).mean()
    PSNR_org_vs_fakeOrg_std     = np.array(test_metrics['PSNR_org_vs_fakeOrg']).std()

    test_metrics['name'].append('mean')
    test_metrics['EPE_org_vs_sim'].append(EPE_org_vs_sim_mean)
    test_metrics['EPE_org_vs_fakeOrg'].append(EPE_org_vs_fakeOrg_mean)
    test_metrics['PSNR_org_vs_sim'].append(PSNR_org_vs_sim_mean)
    test_metrics['PSNR_org_vs_fakeOrg'].append(PSNR_org_vs_fakeOrg_mean)

    test_metrics['name'].append('std')
    test_metrics['EPE_org_vs_sim'].append(EPE_org_vs_sim_std)
    test_metrics['EPE_org_vs_fakeOrg'].append(EPE_org_vs_fakeOrg_std)
    test_metrics['PSNR_org_vs_sim'].append(PSNR_org_vs_sim_std)
    test_metrics['PSNR_org_vs_fakeOrg'].append(PSNR_org_vs_fakeOrg_std)


    df = pd.DataFrame(data=test_metrics)
    pres = os.path.join(p.PATH_RES, 'evaluation')
    check_dir(pres)
    df.to_csv(os.path.join(pres, set + '.csv'))

    return EPE_org_vs_sim_mean, EPE_org_vs_sim_std, \
           EPE_org_vs_fakeOrg_mean, EPE_org_vs_fakeOrg_std, \
           PSNR_org_vs_sim_mean, PSNR_org_vs_sim_std, \
           PSNR_org_vs_fakeOrg_mean, PSNR_org_vs_fakeOrg_std,\
