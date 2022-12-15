import package_utils.motion_handler as pufl
import package_utils.loader         as pul
import package_utils.reader         as pur
import matplotlib.pyplot            as plt
import numpy                        as np

import os

# ----------------------------------------------------------------------------------------------------------------------
def mk_figure(fig, ax, data, fontsize=12, xlabel=False, ylabel=False, key='I', title='?'):

    if 'motion' in key:
        I = ax.imshow(data, cmap='hsv')
    elif '-' not in key:
        I = ax.imshow(data, cmap='gray')
    else:
        I = ax.imshow(data, cmap='hot')

    ax.locator_params(nbins=3)
    if xlabel is not False:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    if ylabel is not False:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=fontsize)

    plt.colorbar(I, ax=ax)
# ----------------------------------------------------------------------------------------------------------------------
def warpped_showcase(I1, I2, OF, Z_warpped, pres, roi=[10, -10, 10, -10]):

    data_to_plot = {
        'I1': I1[roi[0]:roi[1], roi[2]:roi[3]],
        'I2': I2[roi[0]:roi[1], roi[2]:roi[3]],
        'I2-I1': (I2-I1)[roi[0]:roi[1], roi[2]:roi[3]],
        'I2w-I1': (Z_warpped - I1)[roi[0]:roi[1], roi[2]:roi[3]],
        'I2w-I2': (Z_warpped - I2)[roi[0]:roi[1], roi[2]:roi[3]],
        'motion': np.linalg.norm(OF, axis=2)
    }

    fig, axs = plt.subplots(3, 2, layout="constrained")
    keys= list(data_to_plot.keys())
    incr = 0
    
    for ax in axs.flat:
        data = data_to_plot[keys[incr]]
        mk_figure(fig, ax, data, xlabel=False, ylabel=False, key=keys[incr], title=keys[incr])
        incr += 1
    #
    fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2, wspace=0.2)
    plt.savefig(pres, dpi=600)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def warpped_full_image():
    pseq = '/home/laine/Desktop/MotionEstimationDataBaseAnalysis/motion_analysis/PREPARED_DATA/sample_grid/images-sample_grid.pkl'
    pmotion = '/home/laine/Desktop/MotionEstimationDataBaseAnalysis/motion_analysis/PREPARED_DATA/sample_grid/displacement_field-sample_grid.pkl'
    pcf = '/home/laine/Desktop/MotionEstimationDataBaseAnalysis/motion_analysis/PREPARED_DATA/sample_grid/CF-sample_grid.txt'

    pres = '/home/laine/Desktop/diff.png'

    seq = pul.load_pickle(pseq)
    motion = pul.load_pickle(pmotion)
    cf = pur.read_txt(pcf)

    warpped = pufl.warpper(motion[..., 0], seq[..., 0])
    warpped_showcase(seq[..., 0], seq[..., 1], motion[..., 0], warpped, pres, roi=[30, -30, 30, -30])

# ----------------------------------------------------------------------------------------------------------------------
def warpped_patches():

    pdata = '/home/laine/Documents/PROJECTS_IO/CARODEEPMOTION/DATABASEMOTION/IMAGETNET_SAMPLE'
    list_sub_str = os.listdir(pdata)
    if 'backup_parameters' in list_sub_str:
        list_sub_str.remove('backup_parameters')

    for iname in list_sub_str:

        idx_name = os.listdir(os.path.join(os.path.join(pdata, iname)))
        for idx in idx_name:
            patch_name = os.listdir(os.path.join(pdata, iname, idx, 'I1'))

            for key in patch_name:

                pseq0 = os.path.join(pdata, iname, idx, 'I1', key) # + key #'/home/laine/Desktop/DATABASE_MOTION/sample_grid/id_001/I1/' + key
                pseq1 = os.path.join(pdata, iname, idx, 'I2', key)
                pmotion = os.path.join(pdata, iname, idx, 'OF', key) # '/home/laine/Desktop/DATABASE_MOTION/sample_grid/id_001/OF/' + key
                pcf = os.path.join(pdata, iname, idx, 'CF.txt')

                pres = '/home/laine/Desktop/motion_analysis/' + iname + '_' + idx + '_' + key + '.png'

                seq0 = pul.load_pickle(pseq0)
                seq1 = pul.load_pickle(pseq1)
                motion = pul.load_pickle(pmotion)

                seq0 = np.expand_dims(seq0, axis=2)
                seq1 = np.expand_dims(seq1, axis=2)
                seq = np.concatenate((seq0, seq1), axis=2)

                motion = np.expand_dims(motion, axis=3)

                # cf = pur.read_txt(pcf)
                # cf = float(cf[0].split(' ')[-1].split('\n')[0])
                warpped = pufl.warpper(motion[..., 0], seq[..., 0])
                warpped_showcase(seq[..., 0], seq[..., 1], motion[..., 0], warpped, pres, roi=[30, -30, 30, -30])

# ----------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':

    warpped_patches()

# ----------------------------------------------------------------------------------------------------------------------