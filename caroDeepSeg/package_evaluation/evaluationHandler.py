import os
import sys

import numpy                            as np
import matplotlib.pyplot                as plt
from scipy                              import interpolate

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
        if id == 0:
            mae_LI = arr[patient]['LI']
            mae_MA = arr[patient]['MA']
            mae_IMT = arr[patient]['IMT']
        else:
            mae_LI = np.concatenate((mae_LI, arr[patient]['LI']))
            mae_MA = np.concatenate((mae_MA, arr[patient]['MA']))
            mae_IMT = np.concatenate((mae_IMT, arr[patient]['IMT']))
    display(mae_IMT, mae_LI, mae_MA)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def compute_diff(arr1, arr2, CF):
    diff = {}

    for patient in list(arr1.keys()):
        if patient in arr2.keys():
            # print(patient)
            diff[patient] = {}
            idx = compute_intersection(arr1[patient]['LI_id'], arr2[patient]['LI_id'], arr1[patient]['MA_id'], arr2[patient]['MA_id'])
            debug = False
            if debug:
                plt.figure()
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
def get_annotation(path):

    # --- get patient names
    annotation = {}
    patients = os.listdir(path)
    LI_patients = [LIname for LIname in patients if '-LI.txt' in LIname]
    MA_patients = [MAname for MAname in patients if '-MA.txt' in MAname]

    LI_patients.sort()
    MA_patients.sort()
    patients = [patient.split('-')[0] for patient in MA_patients]
    path_contours = []
    for id in range(len(LI_patients)):
        path_contours.append([LI_patients[id], MA_patients[id]])

    # for id, pname in enumerate(path_contours):
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
            self.annotation_methods[key] = get_annotation(param.PMETHODS[key])

        self.A1bis_annotation = get_annotation(param.PA1BIS)
        self.A1_annotation = get_annotation(param.PA1)
        self.A2_annotation = get_annotation(param.PA2)

        self.diff = {}

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

        self.diff[opponent + '_vs_A1'] = compute_diff(self.A1_annotation, array_opponent, self.CF)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    def get_MAE(self, opponent):
        if opponent == 'A1bis':
            array = self.diff[opponent + '_vs_A1']
        elif opponent == 'A2':
            array = self.diff[opponent + '_vs_A1']
        else:
            if opponent in self.annotation_methods.keys():
                array = self.diff[opponent + '_vs_A1']
            else:
                print("Problem with key.\n")
                sys.exit(1)

        self.diff[opponent + '_vs_A1'] = compute_mae(array)