import os
import sys
import csv

import numpy                            as np
import matplotlib.pyplot                as plt
from scipy                              import interpolate

OUTLIER = 500
REMOVE_OUTLIER = True

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def set_axis_style(ax, labels, rot):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=rot)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def mk_violinplot(data, labels, pos_labels, title, x_label, y_label, path, ninfo):

    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), sharey=True)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharey=True)
    ax1.set_title(title)
    parts = ax1.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('blue')
        pc.set_alpha(0.8)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # --- add position of the minimal value
    mean_median = np.min(medians)
    index = np.where(medians == mean_median)[0] + 1
    plt.axhline(y=mean_median, color='g', linestyle='--', linewidth=0.5)
    for id in range(index.shape[0]):
        plt.axvline(x=index[id], color='g', linestyle='--', linewidth=0.5)

    # --- set style for the axes
    set_axis_style(ax=ax1, labels=labels, rot=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # --- save figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, ninfo + title + '.png'), dpi=800)
    plt.close()

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
def display(mae_IMT, mae_LI, mae_MA):

    print('MAE intima-media thickness +/- std: \n')
    print("%.2f" % float(mae_IMT.mean()*1e6), end = ' ')
    print(' +/- ', end = ' ')
    print("%.2f" % float(mae_IMT.std() * 1e6), end = ' ')
    print(' um. \n')
    print('MAE lumen-intima +/- std: \n')
    print("%.2f" % float(mae_LI.mean()*1e6), end = ' ')
    print(' +/- ', end = ' ')
    print("%.2f" % float(mae_LI.std() * 1e6), end = ' ')
    print(' um. \n')
    print('MAE media-adventice +/- std: \n')
    print("%.2f" % float(mae_MA.mean()*1e6), end = ' ')
    print(' +/- ', end = ' ')
    print("%.2f" % float(mae_MA.std() * 1e6), end = ' ')
    print(' um. \n')
    print('############################# \n \n')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_mae(arr):

    mae_LI = np.zeros([])
    mae_MA = np.zeros([])
    mae_IMT = np.zeros([])
    for id, patient in enumerate(arr.keys()):
        if arr[patient]['LI'] is not None:
            if id == 0:
                mae_LI = arr[patient]['LI']
                mae_MA = arr[patient]['MA']
                mae_IMT = arr[patient]['IMT']
            else:
                mae_LI = np.concatenate((mae_LI, arr[patient]['LI']))
                mae_MA = np.concatenate((mae_MA, arr[patient]['MA']))
                mae_IMT = np.concatenate((mae_IMT, arr[patient]['IMT']))

    return {'mae_LI': mae_LI, 'mae_MA': mae_MA, 'mae_IMT': mae_IMT}

    # display(mae_IMT, mae_LI, mae_MA)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_diff(arr1, arr2, CF, common_support):
    diff = {}

    for patient in list(common_support.keys()):
    # for patient in list(arr1.keys()):
        if patient in arr2.keys():
            # print(patient)
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
                diff[patient]['IMT']=None
                diff[patient]['MA'] = None
                diff[patient]['LI'] = None

    return diff

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

    arr1 = list(arr1)
    arr2 = list(arr2)
    arr3 = list(arr3)
    arr4 = list(arr4)

    arr = [arr1, arr2, arr3, arr4]
    intersection = set.intersection(*map(set, arr))
    intersection = np.array(list(intersection))
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

        self.diff = {}
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

        self.diff[opponent + 'VSA1'] = compute_diff(self.A1_annotation, array_opponent, self.CF, self.common_support)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_MAE(self, opponent):
        if opponent == 'A1bis':
            array = self.diff[opponent + 'VSA1']
        elif opponent == 'A2':
            array = self.diff[opponent + 'VSA1']
        else:
            if opponent in self.annotation_methods.keys():
                array = self.diff[opponent + 'VSA1']
            else:
                print("Problem with key.\n")
                sys.exit(1)

        self.metrics[opponent + 'VSA1'] = compute_mae(array)

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
        else:
            out = np.array(intersection)

        return out

    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def write_metrics_to_cvs(self, path, ninfo):

        headers = ['method/expert', 'mae LI (mean)', 'mae LI (std)', 'mae MA (mean)', 'mae MA (std)', 'mae IMT (mean)', 'mae IMT (std)']

        with open(os.path.join(path, ninfo + 'metrics.csv'), 'w') as file:
            # --- create a CSV writer
            writer = csv.writer(file)

            # --- write data to the file
            writer.writerow(headers)
            data = []
            for key in self.metrics.keys():
                data.append([key.split('VS')[0], self.metrics[key]['mae_LI'].mean()*1e6, self.metrics[key]['mae_LI'].std()*1e6, self.metrics[key]['mae_MA'].mean()*1e6, self.metrics[key]['mae_MA'].std()*1e6, self.metrics[key]['mae_IMT'].mean()*1e6, self.metrics[key]['mae_IMT'].std()*1e6])

            for id in range(len(data)):
                writer.writerow(data[id])

        labels = [key.split('VS')[0].replace('Computerized-', '') for key in self.metrics.keys()]
        pos_labels = list(np.linspace(1, len(labels), len(labels)))
        data_LI = [self.metrics[key]['mae_LI']*1e6 for key in self.metrics.keys()]
        data_MA = [self.metrics[key]['mae_MA']*1e6 for key in self.metrics.keys()]
        data_IMT = [self.metrics[key]['mae_IMT']*1e6 for key in self.metrics.keys()]

        # --- reject outlier (error biggers than 500 um)

        id_outlier_LA = [list(np.where(arr > OUTLIER)[0]) for arr in data_LI]
        id_outlier_MA = [list(np.where(arr > OUTLIER)[0]) for arr in data_MA]
        id_outlier_IMT = [list(np.where(arr > OUTLIER)[0]) for arr in data_IMT]

        id_reject = []
        for id in id_outlier_LA:
            id_reject+=id
        for id in id_outlier_MA:
            id_reject+=id
        for id in id_outlier_IMT:
            id_reject += id
        # --- remove duplicate
        id_reject = list(dict.fromkeys(id_reject))
        data_LI = [np.delete(arr, id_reject) for arr in data_LI]
        data_MA = [np.delete(arr, id_reject) for arr in data_MA]
        data_IMT = [np.delete(arr, id_reject) for arr in data_IMT]


        mk_violinplot(data_LI, labels, pos_labels, 'LI', 'method/expert', 'thickness in um', path, ninfo)
        mk_violinplot(data_MA, labels, pos_labels, 'MI', 'method/expert', 'thickness in um', path, ninfo)
        mk_violinplot(data_IMT, labels, pos_labels, 'IMT', 'method/expert', 'thickness in um', path, ninfo)
# ----------------------------------------------------------------------------------------------------------------------------------------------------