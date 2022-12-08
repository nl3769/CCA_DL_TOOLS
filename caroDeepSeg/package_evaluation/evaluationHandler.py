import os
import sys
import csv
import pandas                           as pd
import seaborn                          as sns
import numpy                            as np
import matplotlib.pyplot                as plt
from scipy                              import interpolate

OUTLIER = 500
REMOVE_OUTLIER = True

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def enforce_id_integer(pos, val):

    pos_LI_shift = np.roll(pos, 1)
    diff = pos[1:-1] - pos_LI_shift[1:-1]
    tst = np.in1d(diff, 1)
    count = np.count_nonzero(tst == True)

    if count == (pos.shape[0]-2):
        pos = pos.astype(np.int)
        val = val.astype(np.float)
    else:
        pos_target = np.linspace(int(np.ceil(pos[0])), int(np.floor(pos[-1])), int(np.floor(pos[-1]) - np.ceil(pos[0]) + 1))
        f = interpolate.interp1d(pos, val)
        val = f(pos_target)
        pos = pos_target.astype(np.int)
        val = val.astype(np.float)

    return pos, val

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_diff(arr1, arr2, CF, common_support, opponent):
    out = []
    index = []
    for patient in list(common_support.keys()):
    # for patient in list(arr1.keys()):
        if patient in arr2.keys():
            diff = {}
            diff[patient] = {}
            # idx = compute_intersection(arr1[patient]['LI_id'], arr2[patient]['LI_id'], arr1[patient]['MA_id'], arr2[patient]['MA_id']) # if we want to compare on bigger support but there is no sense do to it for fair comparison
            idx = common_support[patient]
            if idx is not None:
                debug = False
                if debug:
                    plt.figure(1)
                    plt.plot(arr1[patient]['LI_val'])
                    plt.plot(arr2[patient]['LI_val'])
                    plt.show()

                LI_A1_idx = get_idx(arr1[patient]['LI_id'], idx)
                MA_A1_idx = get_idx(arr1[patient]['MA_id'], idx)
                LI_A1bis_idx = get_idx(arr2[patient]['LI_id'], idx)
                MA_A1bis_idx = get_idx(arr2[patient]['MA_id'], idx)

                diff[patient]['LI'] = np.abs((arr1[patient]['LI_val'][LI_A1_idx] - arr2[patient]['LI_val'][LI_A1bis_idx])) * CF[patient]
                diff[patient]['MA'] = np.abs((arr1[patient]['MA_val'][MA_A1_idx] - arr2[patient]['MA_val'][MA_A1bis_idx])) * CF[patient]
                diff[patient]['IMT'] = np.abs(((arr1[patient]['MA_val'][MA_A1_idx] - arr1[patient]['LI_val'][LI_A1_idx]) - (arr2[patient]['MA_val'][MA_A1bis_idx] - arr2[patient]['LI_val'][LI_A1bis_idx]))) * CF[patient]
            else:
                diff[patient]['IMT'] = None
                diff[patient]['MA'] = None
                diff[patient]['LI'] = None

            method_ = opponent.replace('Computerized-', '')
            if diff[patient]['IMT'] is not None:
                for id in range(diff[patient]['IMT'].shape[0]):
                    out += [{'Method': method_, 'Interest': 'IMT', 'Patient': patient, 'Value': diff[patient]['IMT'][id]*1e6}]
                    index += ['IMT']
                    out += [{'Method': method_, 'Interest': 'LI', 'Patient': patient, 'Value': diff[patient]['LI'][id]*1e6}]
                    index += ['LI']
                    out += [{'Method': method_, 'Interest': 'MA', 'Patient': patient, 'Value': diff[patient]['MA'][id]*1e6}]
                    index  += ['MA']
            else:
                out += [{'Method': method_, 'Interest': 'IMT', 'Patient': patient, 'Value': None}]
                index  += ['IMT']
                out += [{'Method': method_, 'Interest': 'LI', 'Patient': patient, 'Value': None}]
                index  += ['LI']
                out += [{'Method': method_, 'Interest': 'MA', 'Patient': patient, 'Value': None}]
                index += ['MA']

    return out, index

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_idx(array, val):

    val = val.flatten()
    array = array.flatten()
    array = np.unique(array)
    idx = np.where(np.in1d(array, val[1:-1]))[0]
    idx = np.unique(idx)

    return idx

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_intersection(arr1, arr2, arr3, arr4):

    arr = [list(arr1), list(arr2), list(arr3), list(arr4)]

    intersection = set.intersection(*map(set, arr))
    intersection = list(intersection)
    intersection.sort()
    intersection = np.array(intersection)

    return intersection

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_CF(path):
    # --- get patient names
    CF = {}
    patients = os.listdir(path)
    patients.sort()

    for id, patient in enumerate(patients):
        path_ = os.path.join(path, patient)
        with open(path_, 'r') as f:
            CF_val = f.readline()
            CF_val = float(CF_val.split(' ')[0]) * 1e-3 # -> convert in meter
            CF[patient.split('_CF')[0]] = CF_val

    return CF

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def get_annotation(path, set):

    # --- get patient names
    annotation = {}
    patients = os.listdir(path)

    if set == 'clin':
        LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname and 'clin' in LIname]
        MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname and 'clin' in MAname]
    elif set == 'tech':
        LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname and 'tech' in LIname]
        MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname and 'tech' in MAname]
    else:
        LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname]
        MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname]

    # --- ensure that we have the same number of files
    for test in LI_patients:
        if test.replace('LI', 'MA') not in MA_patients:
            print("Problem with file name (fuction get_annotation)")
            sys.exit(1)

    LI_patients.sort()
    MA_patients.sort()
    patients = [patient.split('-')[0] for patient in MA_patients]
    path_contours = []
    for id in range(len(LI_patients)):
        path_contours.append([LI_patients[id], MA_patients[id]])

    for id, pname in enumerate(path_contours):
        annotation[patients[id]] = {}
        pLI = os.path.join(path, pname[0])
        pMA = os.path.join(path, pname[1])
        with open(pLI, 'r') as f:
            LI = f.readlines()
        with open(pMA, 'r') as f:
            MA = f.readlines()

        if len(LI) == 2 and len(LI) == 2:
            LI_id = LI[0].split(' ')
            LI_id = LI_id[:-1]
            LI_val = LI[1].split(' ')[:-1]

            MA_id = MA[0].split(' ')
            MA_id = MA_id[:-1]
            MA_val = MA[1].split(' ')[:-1]

        else:
            LI = [key.replace(' \n', '') for key in LI]
            LI_val = [key.split(' ')[1] for key in LI]
            LI_id = [key.split(' ')[0] for key in LI]
            MA = [key.replace(' \n', '') for key in MA]
            MA_val = [key.split(' ')[1] for key in MA]
            MA_id = [key.split(' ')[0] for key in MA]

        # --- ensure that indexes are integer
        [pos_LI, val_LI] = enforce_id_integer(np.array(LI_id).astype(np.float), np.array(LI_val).astype(np.float))
        [pos_MA, val_MA] = enforce_id_integer(np.array(MA_id).astype(np.float), np.array(MA_val).astype(np.float))

        annotation[patients[id]]['LI_id'] = pos_LI
        annotation[patients[id]]['LI_val'] = val_LI
        annotation[patients[id]]['MA_id'] = pos_MA
        annotation[patients[id]]['MA_val'] = val_MA

    return annotation

# ----------------------------------------------------------------------------------------------------------------------------------------------------
class evaluationHandler():

    def __init__(self, param):

        self.annotation_methods = {}
        self.CF = get_CF(param.PCF)

        for key in param.PMETHODS.keys():
            print(key)
            self.annotation_methods[key] = get_annotation(param.PMETHODS[key], param.SET)

        self.A1bis_annotation = get_annotation(param.PA1BIS, param.SET)
        self.A1_annotation = get_annotation(param.PA1, param.SET)
        self.A2_annotation = get_annotation(param.PA2, param.SET)
        self.common_support = self.get_common_support()

        self.data_frame = {}

        self.diff = []
        self.index = []
        self.df_diff = []
        self.metrics = {}

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_diff(self, opponent):
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

        [diff_, index_] = compute_diff(self.A1_annotation, array_opponent, self.CF, self.common_support, opponent)
        self.diff += diff_
        self.index += index_

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_common_support(self):

        """ Compute common support, mean position where IMC was segmented by each center. """

        common_support = {}

        # --- we first retreive the common processed patients
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

        self.df_diff = pd.DataFrame(self.diff, index=self.index)

        mae_IMT = self.df_diff.filter(like='IMT', axis=0).groupby(['Method']).mean()
        mae_IMT.rename(columns={'Value': 'mean IMT'}, inplace=True)
        std_IMT = self.df_diff.filter(like='IMT', axis=0).groupby(['Method']).std()
        std_IMT.rename(columns={'Value': 'std IMT'}, inplace=True)

        mae_LI = self.df_diff.filter(like='LI', axis=0).groupby(['Method']).mean()
        mae_LI.rename(columns={'Value': 'mean LI'}, inplace=True)
        std_LI = self.df_diff.filter(like='LI', axis=0).groupby(['Method']).std()
        std_LI.rename(columns={'Value': 'std LI'}, inplace=True)

        mae_LI = self.df_diff.filter(like='MA', axis=0).groupby(['Method']).mean()
        mae_LI.rename(columns={'Value': 'mean MA'}, inplace=True)
        std_MA = self.df_diff.filter(like='MA', axis=0).groupby(['Method']).std()
        std_MA.rename(columns={'Value': 'std MA'}, inplace=True)

        metrics_df = pd.concat([mae_IMT, mae_LI, mae_LI, std_IMT, std_LI, std_LI], axis=1)
        metrics_df.to_csv(os.path.join(path, ninfo + 'metrics.csv'), index=True, header=True)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def mk_plot_seaborn(self, path, ninfo):

        # BOX PLOT
        # --- plot all mae (IMT, LI, MA) on the same graph
        sns.boxplot(self.df_diff, x='Method', y='Value', showfliers=False, hue='Interest')
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'boxplot_full' + '.png'), dpi=500)
        plt.close()

        # --- plot MA mae
        df = self.df_diff[self.df_diff['Interest'] == 'MA']
        sns.boxplot(df, x='Method', y='Value', showfliers=False)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        mean_val = df.groupby(['Method']).median()
        min_median = np.min(mean_val.to_numpy().squeeze())
        plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'boxplot_MA' + '.png'), dpi=500)
        plt.close()

        # --- plot LI mae
        df = self.df_diff[self.df_diff['Interest'] == 'LI']
        sns.boxplot(df, x='Method', y='Value', showfliers=False)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        mean_val = df.groupby(['Method']).median()
        min_median = np.min(mean_val.to_numpy().squeeze())
        plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'boxplot_LI' + '.png'), dpi=500)
        plt.close()

        # --- plot IMT mae
        df = self.df_diff[self.df_diff['Interest'] == 'IMT']
        sns.boxplot(df, x='Method', y='Value', showfliers=False)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        mean_val = df.groupby(['Method']).median()
        min_median = np.min(mean_val.to_numpy().squeeze())
        plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'boxplot_IMT' + '.png'), dpi=500)
        plt.close()

        # VIOLIN PLOT
        df_rm_outliers = self.df_diff[self.df_diff['Value'] < 600]
        # --- plot all mae (IMT, LI, MA) on the same graph

        sns.violinplot(df_rm_outliers, x='Method', y='Value', showfliers=False, hue='Interest', linewidth=0.2)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'violin_full' + '.png'), dpi=500)
        plt.close()

        # --- plot MA mae
        df = df_rm_outliers[df_rm_outliers['Interest'] == 'MA']
        sns.violinplot(df, x='Method', y='Value', showfliers=False, linewidth=0.8)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        mean_val = df.groupby(['Method']).median()
        min_median = np.min(mean_val.to_numpy().squeeze())
        plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'violin_MA' + '.png'), dpi=500)
        plt.close()

        # --- plot LI mae
        df = df_rm_outliers[df_rm_outliers['Interest'] == 'LI']
        sns.violinplot(df, x='Method', y='Value', showfliers=False, linewidth=0.8)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        mean_val = df.groupby(['Method']).median()
        min_median = np.min(mean_val.to_numpy().squeeze())
        plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'violin_LI' + '.png'), dpi=500)
        plt.close()

        # --- plot IMT mae
        df = df_rm_outliers[df_rm_outliers['Interest'] == 'IMT']
        sns.violinplot(df, x='Method', y='Value', showfliers=False, linewidth=0.8)
        sns.despine()
        plt.xticks(rotation=30, ha='right')
        mean_val = df.groupby(['Method']).median()
        min_median = np.min(mean_val.to_numpy().squeeze())
        plt.axhline(y=min_median, color='r', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(path, ninfo + 'violin_IMT' + '.png'), dpi=500)
        plt.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def write_unprocessed_images(self, path):

        # --- get key from A1 expert (suppose to contain all processed images)
        unprocessed_patients = list(self.A1_annotation.keys())
        processed_patients = []
        for processed_patient in self.common_support.keys():
            if self.common_support[processed_patient] is not None:
                processed_patients.append(processed_patient)
                unprocessed_patients.remove(processed_patient)

        with open(os.path.join(path, 'unprocessed.txt'), 'w') as f:
            for patient in unprocessed_patients:
                f.write(patient + '\n')

# ----------------------------------------------------------------------------------------------------------------------------------------------------