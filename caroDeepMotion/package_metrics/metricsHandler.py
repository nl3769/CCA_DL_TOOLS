import os
import torch

import matplotlib.pyplot        as plt
import nibabel                  as nib
import numpy                    as np
import pandas                   as pd
import seaborn                  as sns
import package_network.utils    as pnu
import package_utils.loader     as pul

from tqdm                       import tqdm
from abc                        import abstractmethod

class evaluation():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, save_metrics, nmethod):

        self.nmethod = nmethod
        self.save_metrics = save_metrics
        self.epe = []
        self.angle = []
        self.index = []

    # ------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def read_cf(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def get_patients(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def load_data(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def compute_EPE(self, pnames):

        # --- computer EPE: epe = mean((||x_pred, y_pred||_2 - ||x_ref, y_ref||_2))
        for patient in pnames:

            ref = self.res_ref[patient]
            ref[..., 0] = ref[..., 0] * self.CF[patient]['xCF']
            ref[..., 1] = ref[..., 1] * self.CF[patient]['zCF']

            pred = self.res_pred[patient]
            pred[..., 0] = pred[..., 0] * self.CF[patient]['xCF']
            pred[..., 1] = pred[..., 1] * self.CF[patient]['zCF']

            diff = ref - pred
            epe_abs = np.linalg.norm(diff, axis=2)
            epe_rel = np.linalg.norm(diff, axis=2) / np.linalg.norm(ref, axis=2)

            self.epe += [{'epe abs': np.mean(epe_abs)*1e6, 'epe rel': np.mean(epe_rel)*1e2, 'Method': self.nmethod}]
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
    def compute_angle_error(self, pnames):

        # --- computer EPE: epe = mean((||x_pred, y_pred||_2 - ||x_ref, y_ref||_2))
        for patient in pnames:

            ref = self.res_ref[patient]
            pred = self.res_pred[patient]
            ones_ = np.ones(pred.shape[:-1])
            ones_ = np.expand_dims(ones_, axis=2)
            norm_ref = np.linalg.norm(np.concatenate((ref, ones_), axis=2), axis=2)      # add 1 to avoid 0-division
            norm_pred = np.linalg.norm(np.concatenate((pred, ones_), axis=2), axis=2)    # add 1 to avoid 0-division
            angle_pred = np.rad2deg(np.arccos((pred[..., 0]) / np.linalg.norm(pred, axis=2)))
            angle_ref = np.rad2deg(np.arccos((ref[..., 0]) / np.linalg.norm(ref, axis=2)))

            AE = np.arccos((np.multiply(pred[..., 0], ref[..., 0]) + np.multiply(pred[..., 1], ref[..., 1]) + 1) / (np.multiply(norm_ref, norm_pred)))
            AE = np.rad2deg(AE)
            # self.angle+= [{'phase': np.mean(np.abs(angle_pred - angle_ref)), 'Method': 'DG'}]
            self.angle += [{'AE': np.mean(AE), 'Method': self.nmethod}]
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

# ----------------------------------------------------------------------------------------------------------------------
class evaluationDG(evaluation):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, pres, save_metrics, DG):
        evaluation.__init__(self, save_metrics, DG)
        self.patients = self.get_patients(pres, 'num_res')
        self.res_pred, self.res_ref, self.grid = self.load_data()
        self.CF = self.read_cf()

    # ------------------------------------------------------------------------------------------------------------------
    def read_cf(self):
        cf_val = {}
        for patient in self.patients.keys():
            with open(self.patients[patient]['cf'], 'r') as f:
                cf =f.read()

            cf_val[patient] = {'xCF': float(cf), 'zCF': float(cf)}

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
        """Load results. """

        res_pred = {}
        res_ref = {}
        grid = {}

        for patient in self.patients.keys():
            res_pred[patient] = np.array(nib.load(self.patients[patient]['pred']).get_data())
            res_ref[patient] = np.array(nib.load(self.patients[patient]['ref']).get_data())
            grid[patient] = np.array(nib.load(self.patients[patient]['grid']).get_data())

        return res_pred, res_ref, grid

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.compute_angle_error(self.patients.keys())
        self.compute_EPE(self.patients.keys())
        self.plot_metrics()

# ----------------------------------------------------------------------------------------------------------------------
class evaluationDL(evaluation):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, save_metrics, pdata, psplit, p):

        evaluation.__init__(self, save_metrics, p.METHOD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patients = self.get_patients(psplit, pdata)
        self.netEncoder, self.netFlow = self.load_model(p)

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self, p):

        netEncoder, netFlow = pnu.load_model_flow(p)
        netEncoder = netEncoder.to(self.device)
        netFlow = netFlow.to(self.device)

        return netEncoder, netFlow

    # ------------------------------------------------------------------------------------------------------------------
    def read_cf(self):
        cf_val = {}
        for patient in self.patients.keys():
            with open(self.patients[patient]['cf'], 'r') as f:
                cf =f.read()

            cf_val[patient] = float(cf)

        return cf_val

    # ------------------------------------------------------------------------------------------------------------------
    def get_patients(self, npatient, pdata):

        # --- list patients
        with open(npatient, 'r') as f:
            list_patients = f.readlines()
        patients = [os.path.join(pdata, patient.split('\n')[0]) for patient in list_patients]
        lst_of, lst_I1, lst_I2, lst_cf = [], [], [], []
        for patient in patients:
            folders = os.listdir(patient)
            for folder in folders:
                # --- list files
                of_ = os.listdir(os.path.join(patient, folder, 'OF'))
                I1_ = os.listdir(os.path.join(patient, folder, 'I1'))
                I2_ = os.listdir(os.path.join(patient, folder, 'I2'))
                cf_ = os.path.join(patient, folder, 'CF.txt')
                cf_ = [cf_] * len(I1_)
                # --- sort file name
                of_.sort(), I1_.sort(), I2_.sort()
                # --- add path
                lst_of += [os.path.join(patient, folder, 'OF', key) for key in of_]
                lst_I1 += [os.path.join(patient, folder, 'I1', key) for key in I1_]
                lst_I2 += [os.path.join(patient, folder, 'I2', key) for key in I2_]
                lst_cf += cf_

        patients = {
            'pI1': lst_I1,
            'pI2': lst_I2,
            'pof': lst_of,
            'pcf': lst_cf}

        return patients

    # ------------------------------------------------------------------------------------------------------------------
    def load_data(self, nI1, nI2, nOF, nCF):

        I1, I2, OF, CF = [], [], [], []
        for path in zip(nI1, nI2, nOF, nCF):
            I1_ = pul.load_pickle(path[0])
            I1_ -= np.min(I1_)
            I1_ /= np.max(I1_)
            I1.append(I1_[None, ...])

            I2_ = pul.load_pickle(path[1])
            I2_ -= np.min(I2_)
            I2_ /= np.max(I2_)
            I2.append(I2_[None, ...])

            OF.append(pul.load_pickle(path[2]))

            with open(path[3], 'r') as f:
                data = f.readlines()
            data = [key.split('\n')[0] for key in data]
            cf_ = {
                'xCF': float(data[0].split(" ")[-1]),
                'zCF': float(data[1].split(" ")[-1])}

            CF.append(cf_)

        I1 = np.array(I1)
        I2 = np.array(I2)
        OF = np.array(OF)

        return I1, I2, OF, CF

    # ------------------------------------------------------------------------------------------------------------------
    def inference(self, I1, I2):

        I1 = torch.from_numpy(I1).to(self.device)
        I2 = torch.from_numpy(I2).to(self.device)
        mask = torch.ones(I1.shape).to(self.device)

        fmap1, skc1, fmap2, skc2 = self.netEncoder(I1, I2)
        flow_pred = self.netFlow(I1, fmap1, fmap2, mask)
        flow_pred = np.array(flow_pred[-1].detach().to('cpu'))

        return flow_pred

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        # --- prediction by batch
        batch_size = 4
        nb_patients = len(self.patients['pI1'])
        nb_loop = int(np.ceil(nb_patients/batch_size))

        self.res_ref = {}
        self.res_pred = {}
        self.CF = {}

        for id_loop in tqdm(range(nb_loop)):

            if id_loop == (nb_loop-1):
                nI1 = self.patients['pI1'][id_loop * batch_size:-1]
                nI2 = self.patients['pI2'][id_loop * batch_size:-1]
                nOF = self.patients['pof'][id_loop * batch_size:-1]
                nCF = self.patients['pcf'][id_loop * batch_size:-1]
            else:
                nI1 = self.patients['pI1'][id_loop*batch_size:id_loop*batch_size + batch_size]
                nI2 = self.patients['pI2'][id_loop*batch_size:id_loop*batch_size + batch_size]
                nOF = self.patients['pof'][id_loop*batch_size:id_loop*batch_size + batch_size]
                nCF = self.patients['pcf'][id_loop*batch_size:id_loop*batch_size + batch_size]

            I1, I2, OF, CF = self.load_data(nI1, nI2, nOF, nCF)
            pred = self.inference(I1, I2)

            # --- fill dictionary to fit with the others functions
            loop_size = batch_size if id_loop < (nb_loop - 1) else len(self.patients['pcf']) - id_loop*batch_size - 1
            for id_batch in range(loop_size):
                patient = self.patients['pI1'][id_loop*batch_size + id_batch].split('/')[-4:]
                patient.remove('I1')
                pname = patient[0] + "_" + patient[1] + "_" + patient[2].split(".")[0]
                self.res_ref[pname] = OF[id_batch, ][..., ::2]
                self.res_pred[pname] = np.transpose(pred[id_batch, ], (1, 2, 0))
                self.CF[pname] = CF[id_batch]

        self.compute_angle_error(pnames=self.res_pred.keys())
        self.compute_EPE(pnames=self.res_pred.keys())
        self.plot_metrics()

# ----------------------------------------------------------------------------------------------------------------------
class evaluationDLFullImg(evaluation):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, pres, save_metrics, nmethod):
        evaluation.__init__(self, save_metrics, nmethod)
        self.patients = self.get_patients(pres)
        self.res_pred, self.res_ref = self.load_data()
        self.CF = self.read_cf()

    # ------------------------------------------------------------------------------------------------------------------
    def read_cf(self):
        cf_val = {}
        for patient in self.patients.keys():
            with open(self.patients[patient]['cf'], 'r') as f:
                cf =f.read()

            cf_val[patient] = {
                "xCF": float(cf.split('\n')[0].split(':')[-1]),
                "zCF": float(cf.split('\n')[1].split(':')[-1])}

        return cf_val

    # ------------------------------------------------------------------------------------------------------------------
    def get_patients(self, path):
        # --- list patients
        list_patients = os.listdir(path)
        list_patients.sort()
        patients = {}

        for patient in list_patients:
            ndata = os.listdir(os.path.join(path, patient))
            ndata_pred = [key for key in ndata if 'pred' in key][0]
            ndata_ref = [key for key in ndata if 'gt' in key][0]
            cf = [key for key in ndata if 'cf' in key][0]

            patients[patient] = {}
            patients[patient]['pred'] = os.path.join(path, patient, ndata_pred)
            patients[patient]['ref'] = os.path.join(path, patient, ndata_ref)
            patients[patient]['cf'] = os.path.join(path, patient, cf)

        return patients

    # ------------------------------------------------------------------------------------------------------------------
    def load_data(self):

        res_pred = {}
        res_ref = {}

        for patient in self.patients.keys():
            res_pred[patient] = np.transpose(pul.load_pickle(self.patients[patient]['pred']), (1, 2, 0))
            res_ref[patient] = np.transpose(pul.load_pickle(self.patients[patient]['ref']), (1, 2, 0))

        return res_pred, res_ref

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.compute_angle_error(self.patients.keys())
        self.compute_EPE(self.patients.keys())
        self.plot_metrics()

# ----------------------------------------------------------------------------------------------------------------------