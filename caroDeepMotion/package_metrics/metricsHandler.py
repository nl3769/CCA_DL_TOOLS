import os

import matplotlib.pyplot    as plt
import nibabel              as nib
import numpy                as np
import pandas               as pd
import seaborn              as sns

class evaluation():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, pres, save_metrics):

        subres = 'num_res'
        self.patients = self.get_patients(pres, subres)
        self.save_metrics = save_metrics
        self.pres = pres
        self.res_pred, self.res_ref, self.grid = self.load_data()
        self.CF = self.read_cf()
        self.epe = []
        self.angle = []
        self.index = []

    # ------------------------------------------------------------------------------------------------------------------
    def read_cf(self):
        cf_val = {}
        for patient in self.patients.keys():
            with open(self.patients[patient]['cf'], 'r') as f:
                cf =f.read()

            cf_val[patient] = float(cf)

        return cf_val

    # ------------------------------------------------------------------------------------------------------------------
    def get_patients(self, path, subres):
        # --- list patients
        list_patients = os.listdir(path)
        patients = {}

        for patient in list_patients:
            ndata = os.listdir(os.path.join(path, patient, subres))
            ndata_pred = [key for key in ndata if 'pred' in key][0]
            ndata_ref = [key for key in ndata if 'ref' in key][0]
            ndata_grid = [key for key in ndata if 'grid' in key][0]
            cf = [key for key in ndata if 'cf' in key][0]

            patients[patient] = {}
            patients[patient]['pred'] = os.path.join(path, patient, subres, ndata_pred)
            patients[patient]['ref'] = os.path.join(path, patient, subres, ndata_ref)
            patients[patient]['grid'] = os.path.join(path, patient, subres, ndata_grid)
            patients[patient]['cf'] = os.path.join(path, patient, subres, cf)

        return patients

    # ------------------------------------------------------------------------------------------------------------------
    def load_data(self):

        res_pred = {}
        res_ref = {}
        grid = {}

        for patient in self.patients.keys():
            res_pred[patient] = np.array(nib.load(self.patients[patient]['pred']).get_data())
            res_ref[patient] = np.array(nib.load(self.patients[patient]['ref']).get_data())
            grid[patient] = np.array(nib.load(self.patients[patient]['grid']).get_data())

        return res_pred, res_ref, grid

    # ------------------------------------------------------------------------------------------------------------------
    def compute_EPE(self):

        # --- computer EPE: epe = mean((||x_pred, y_pred||_2 - ||x_ref, y_ref||_2))
        for patient in self.patients.keys():

            ref = self.res_ref[patient] * self.CF[patient]
            pred = self.res_pred[patient] * self.CF[patient]

            diff = ref - pred
            epe_abs = np.linalg.norm(diff, axis=2)
            epe_rel = np.linalg.norm(diff, axis=2) / np.linalg.norm(ref, axis=2)

            self.epe += [{'epe abs': np.mean(epe_abs)*1e6, 'epe rel': np.mean(epe_rel)*1e2, 'Method': 'DG'}]
            self.index += [patient]
            debug = False
            if debug:
                plt.figure(1)
                plt.subplot2grid((1, 3), (0, 0), colspan=1)
                plt.imshow(epe_abs, cmap='jet')
                plt.colorbar()
                plt.title('epe')

                plt.subplot2grid((1, 3), (0, 1), colspan=1)
                plt.imshow(np.linalg.norm(pred, axis=2), cmap='jet')
                plt.colorbar()
                plt.title('pred')

                plt.subplot2grid((1, 3), (0, 2), colspan=1)
                plt.imshow(np.linalg.norm(ref, axis=2), cmap='jet')
                plt.colorbar()
                plt.title('ref')

                plt.tight_layout()
                plt.savefig(os.path.join('/home/laine/Desktop/tmp', patient))
                plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def compute_angle_error(self):

        # --- computer EPE: epe = mean((||x_pred, y_pred||_2 - ||x_ref, y_ref||_2))
        for patient in self.patients.keys():
            print(patient)
            ref = self.res_ref[patient]
            pred = self.res_pred[patient]
            ones_ = np.ones(pred.shape[:-1])
            ones_ = np.expand_dims(ones_, axis=2)
            norm_ref = np.linalg.norm(np.concatenate((ref, ones_), axis=2), axis=2)      # add 1 to avoid 0-division
            norm_pred = np.linalg.norm(np.concatenate((pred, ones_), axis=2), axis=2)    # add 1 to avoid 0-division
            angle_pred = np.rad2deg(np.arccos((pred[..., 0]) / np.linalg.norm(pred, axis=2)))
            angle_ref  = np.rad2deg(np.arccos((ref[..., 0]) / np.linalg.norm(ref, axis=2)))

            AE = np.arccos((np.multiply(pred[..., 0], ref[..., 0]) + np.multiply(pred[..., 1], ref[..., 1]) + 1) / (np.multiply(norm_ref, norm_pred)))
            AE = np.rad2deg(AE)
            # self.angle+= [{'phase': np.mean(np.abs(angle_pred - angle_ref)), 'Method': 'DG'}]
            self.angle += [{'AE': np.mean(AE), 'Method': 'DG'}]

            debug = False

            if debug:
                plt.figure(1)
                plt.subplot2grid((1, 3), (0, 0), colspan=1)
                plt.imshow(np.abs(angle_pred - angle_ref), cmap='jet')
                plt.colorbar()
                plt.title('epe')

                plt.subplot2grid((1, 3), (0, 1), colspan=1)
                plt.imshow(angle_pred, cmap='jet')
                plt.colorbar()
                plt.title('pred')

                plt.subplot2grid((1, 3), (0, 2), colspan=1)
                plt.imshow(angle_ref, cmap='jet')
                plt.colorbar()
                plt.title('ref')

                plt.tight_layout()
                plt.savefig(os.path.join('/home/laine/Desktop/tmp', patient))
                plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_metrics(self):

        dataframe = pd.DataFrame(self.epe, index=self.index)

        # --- display epe
        sns.boxplot(dataframe, x='Method', y='epe abs', showfliers=False)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_metrics, 'boxplot_epe_abs' + '.png'), dpi=500)
        plt.close()

        sns.violinplot(dataframe, x='Method', y='epe abs')
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_metrics, 'violin_epe_abs' + '.png'), dpi=500)
        plt.close()

        # --- display epe
        sns.boxplot(dataframe, x='Method', y='epe rel', showfliers=False)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_metrics, 'boxplot_epe_rel' + '.png'), dpi=500)
        plt.close()

        sns.violinplot(dataframe, x='Method', y='epe rel')
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_metrics, 'violin_epe_rel' + '.png'), dpi=500)
        plt.close()

        # --- display angle
        dataframe = pd.DataFrame(self.angle, index=self.index)
        sns.boxplot(dataframe, x='Method', y='AE', showfliers=False)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_metrics, 'boxplot_angle' + '.png'), dpi=500)
        plt.close()

        sns.violinplot(dataframe, x='Method', y='AE')
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_metrics, 'violin_angle' + '.png'), dpi=500)
        plt.close()
