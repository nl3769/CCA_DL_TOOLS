import os
import sys
import pandas                           as pd
import numpy                            as np
import matplotlib.pyplot                as plt
import package_evaluation.utils         as peu

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class evaluationHandler():

    def __init__(self, param):

        self.annotation_methods = {}
        self.CF = peu.get_CF(param.PCF)

        # --- load data
        for key in param.PMETHODS.keys():
            print(key)
            self.annotation_methods[key] = peu.get_annotation(param.PMETHODS[key], param.SET)

        self.A1bis_annotation = peu.get_annotation(param.PA1BIS, param.SET)
        self.A1_annotation = peu.get_annotation(param.PA1, param.SET)
        self.A2_annotation = peu.get_annotation(param.PA2, param.SET)
        self.common_support = self.get_common_support()

        self.data_frame = {}
        self.diff = []
        self.index = []
        self.metrics = {}
        self.PDM = []
        self.PDM_index = []
        self.hd_index = []
        self.hd_LI = []
        self.hd_MA = []

        self.df_diff = pd.DataFrame()
        self.df_hausdorff_LI = pd.DataFrame()
        self.df_hausdorff_MA = pd.DataFrame()
        self.df_pdm = pd.DataFrame()
        self.df_bias = pd.DataFrame()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_diff(self):

        keys = list(self.annotation_methods.keys()) + ['A1bis', 'A2']

        for opponent in keys:
            if opponent == 'A1bis':
                array_opponent = self.A1bis_annotation
            elif opponent == 'A2':
                array_opponent = self.A2_annotation
            else:
                if opponent in self.annotation_methods.keys():
                    array_opponent = self.annotation_methods[opponent]
                else:
                    print("Problem with key.\n")
                    sys.exit(1)

            [diff_, index_] = peu.compute_diff(self.A1_annotation, array_opponent, self.CF, self.common_support, opponent)
            self.diff += diff_
            self.index += index_

        self.df_diff = pd.DataFrame(self.diff, index=self.index)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_hausdorff(self):

        keys = list(self.annotation_methods.keys()) + ['A1bis', 'A2']
        arr1 = arr = self.A1_annotation
        for key in keys:
            if key == 'A1bis':
                arr2 = self.A1bis_annotation
            elif key == 'A2':
                arr2 = self.A2_annotation
            else:
                if key in self.annotation_methods.keys():
                    arr2 = self.annotation_methods[key]

            key = key.replace('Computerized-', '')
            hd_LI, hd_MA, index = peu.compute_hausdorff_distance(arr1, arr2, self.CF, self.common_support, key)

            self.hd_LI += hd_LI
            self.hd_MA += hd_MA
            self.hd_index += index

        self.df_hausdorff_LI = pd.DataFrame(self.hd_LI, index=self.hd_index)
        self.df_hausdorff_MA = pd.DataFrame(self.hd_MA, index=self.hd_index)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_PDM(self):

        keys = list(self.annotation_methods.keys()) + ['A1', 'A1bis', 'A2']

        for key in keys:
            if key == 'A1bis':
                arr = self.A1bis_annotation
            elif key == 'A1':
                arr = self.A1_annotation
            elif key == 'A2':
                arr = self.A2_annotation
            else:
                if key in self.annotation_methods.keys():
                    arr = self.annotation_methods[key]

            key = key.replace('Computerized-', '')
            DvB1SB2, DvB2SB1, PDM_avg, index = peu.PolyDistMethodHandler(arr, self.CF, self.common_support, key)

            self.PDM += PDM_avg
            self.PDM_index += index

        self.df_pdm = pd.DataFrame(self.PDM, index=self.PDM_index)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_common_support(self):
        """ Compute common support, mean position where IMC was segmented by each center. """

        common_support = {}

        # --- we first retrieve the common processed patients
        for patient in self.A1_annotation.keys():
            condition = True

            if patient not in (self.A1bis_annotation.keys()):
                condition = False
            if patient not in (self.A2_annotation.keys()):
                condition = False

            for method in self.annotation_methods.keys():
                if patient not in (self.annotation_methods[method].keys()):
                    condition = False
                    break

            if condition:
                common_support[patient] = self.get_full_intersection(patient)

        return common_support

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_full_intersection(self, patient):

        # --- get indexes of each annotation (method + experts)
        list_idx = []

        list_idx.append(list(self.A1_annotation[patient]['LI_id']))
        list_idx.append(list(self.A1_annotation[patient]['MA_id']))

        list_idx.append(list(self.A1bis_annotation[patient]['LI_id']))
        list_idx.append(list(self.A1bis_annotation[patient]['MA_id']))

        list_idx.append(list(self.A2_annotation[patient]['LI_id']))
        list_idx.append(list(self.A2_annotation[patient]['MA_id']))

        for method in self.annotation_methods.keys():
            list_idx.append(list(self.annotation_methods[method][patient]['LI_id']))
            list_idx.append(list(self.annotation_methods[method][patient]['MA_id']))

        intersection = list(set.intersection(*map(set, list_idx)))

        if not intersection:
            out = None
            DEBUG = False
            if DEBUG:
                print('A1 LI min: ', self.A1_annotation[patient]['LI_id'][0])
                print('A1 MA min: ', self.A1_annotation[patient]['MA_id'][0])
                print('A1bis LI min: ', self.A1bis_annotation[patient]['LI_id'][0])
                print('A1bis MA min: ', self.A1bis_annotation[patient]['MA_id'][0])
                print('A2 LI min: ', self.A2_annotation[patient]['LI_id'][0])
                print('A2 MA min: ', self.A2_annotation[patient]['MA_id'][0])

                print('A1 LI max: ', self.A1_annotation[patient]['LI_id'][-1])
                print('A1 MA max: ', self.A1_annotation[patient]['MA_id'][-1])
                print('A1bis LI max: ', self.A1bis_annotation[patient]['LI_id'][-1])
                print('A1bis MA max: ', self.A1bis_annotation[patient]['MA_id'][-1])
                print('A2 LI max: ', self.A2_annotation[patient]['LI_id'][-1])
                print('A2 MA max: ', self.A2_annotation[patient]['MA_id'][-1])
        else:
            out = np.array(intersection)

        return out

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def write_metrics_to_cvs(self, path, ninfo):

        # --- HAUSDORFF DISTANCE
        peu.mk_cvs_hd(self.df_hausdorff_LI, self.df_hausdorff_MA, path, ninfo + "_IMT_pdm")

        # --- POLYLINE DISTANCE
        peu.mk_cvs_pdm(self.df_pdm, path, ninfo + "_IMT_pdm")
        peu.mk_cvs_pdm(self.df_bias, path, ninfo + "pdm_IMT_bias_signed")
        peu.mk_cvs_pdm(pd.concat([self.df_bias["Value"].abs(), self.df_bias["Method"]], axis=1), path, ninfo + "pdm_IMT_bias_unsigned")

        # --- MEAN AVERAGE ERROR
        peu.mk_cvs_epe(self.df_diff, path, ninfo)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def compute_bias_pdm(self):

        nmethod = list(self.df_pdm.groupby('Method').nunique().index)
        nref = 'A1'
        df_ref = self.df_pdm[self.df_pdm['Method'] == nref]

        for nmethod_ in nmethod:
            opponent = self.df_pdm[self.df_pdm['Method'] == nmethod_]
            df = pd.concat([pd.DataFrame(df_ref['Value'] - opponent['Value']), pd.DataFrame(opponent["Method"])], axis=1)
            self.df_bias = pd.concat([self.df_bias, df], axis=0)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def mk_plot_seaborn(self, path, ninfo):

        peu.mk_plot_hausdorff(self.df_hausdorff_LI, self.df_hausdorff_MA, path, ninfo + "_hausdorff")

        peu.mk_plot_pdm(self.df_pdm, path, ninfo + "pdm_")
        df_bias = self.df_bias[self.df_bias['Method'] != 'A1']
        peu.mk_plot_pdm(df_bias, path, ninfo + "pdm_bias_signed")
        peu.mk_plot_pdm(pd.concat([df_bias["Value"].abs(), df_bias["Method"]], axis=1), path, ninfo + "pdm_bias_unsigned")

        peu.mk_plot_epe(self.df_diff, path, ninfo)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def write_unprocessed_images(self, path, set):

        # --- get key from A1 expert (suppose to contain all processed images)
        unprocessed_patients = list(self.A1_annotation.keys())
        processed_patients = []
        for processed_patient in self.common_support.keys():
            if self.common_support[processed_patient] is not None:
                processed_patients.append(processed_patient)
                unprocessed_patients.remove(processed_patient)

        with open(os.path.join(path, set + '_unprocessed.txt'), 'w') as f:
            for patient in unprocessed_patients:
                f.write(patient + '\n')

# --------------------------------------------------------------------------------------------------------------------------------------------------------