import os
import random
import numpy as np

import package_debug.visualisation                  as dbv
import package_utils.motion_handler                   as pufh
import package_utils.loader                         as pl
import package_visualization.make_figure            as pvmf

class databaseVisualization():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p):

        self.parameters = p
        self.patient = {}
        self.subfolder = ['I1', 'I2', 'M1', 'M2', 'OF']

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_patient(subfolder: list, pdata: str, nb_val=100):
        """ Get patient to display with random behavior -> patients are randomly selected. """

        patient = {}
        files = os.listdir(pdata)
        fparam = [key for key in files if 'param' in key]
        for key in fparam:
            files.remove(key)
        # --- check dim
        if len(files) < nb_val:
            nb_val = len(files)
        files.sort()
        for id in range(nb_val):

            pres_ = os.path.join(pdata, files[id])
            pname = os.listdir(pres_)

            # --- get random patient
            dim_seq = len(pname)
            seqID = random.randint(1, dim_seq-1)
            pname = pname[seqID]

            # --- get random patch
            pres_ = os.path.join(pres_, pname)
            npatches = os.listdir(os.path.join(pres_, subfolder[0]))
            dim_patches = len(npatches)
            patchID = random.randint(0, dim_patches - 1)
            npatch = npatches[patchID]

            # --- get file name
            patient[files[id]] = {}
            for key in subfolder:

                if "I" in key:
                    patient[files[id]][key] = os.path.join(pres_, key, npatch.split('.')[0] + ".pkl")
                elif "M" in key:
                    patient[files[id]][key] = os.path.join(pres_, key, npatch.split('.')[0] + ".pkl")
                elif "OF" in key:
                    patient[files[id]][key] = os.path.join(pres_, key, npatch.split('.')[0] + ".pkl")

        return patient

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def visualization(patients, subfolder, pres):

        for patient in patients.keys():

            data = {}
            for key in subfolder:

                if "OF" in key:
                    of = pl.load_pickle(patients[patient][key])
                    data[key] = of
                else:
                    data[key] = pl.load_pickle(patients[patient][key])

            xof = data["OF"][..., 0]
            zof = data["OF"][..., 2]
            normOF = np.sqrt(np.power(xof, 2) + np.power(zof, 2))
            argOF = np.arccos(np.divide(xof, normOF))
            argOF = np.rad2deg(argOF)

            I2_warpped = pufh.warpper(data["OF"], data["I1"])

            tmp = patients[patient][key].split('/')[-1].split('.')[0]
            pres_ = os.path.join(pres, patient + "_" + tmp + ".png")
            pvmf.make_figure(data["I1"], data["I2"], data["M1"], data["M2"], I2_warpped, xof, zof, argOF, normOF, pres_)

    # ------------------------------------------------------------------------------------------------------------------

    def __call__(self):

        self.patient = self.get_patient(self.subfolder, self.parameters.PDATA)
        self.visualization(self.patient, self.subfolder, self.parameters.PRES)
