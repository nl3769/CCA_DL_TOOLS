import os
import numpy
import random

import numpy as np

import package_utils.fold_handler as pfh

# ----------------------------------------------------------------------------------------------------------------------
def get_permutation_id(k):

    org = np.linspace(0, k-1, k)
    idx = np.zeros([k, k])
    idx[0, ] = org
    for id in range(k-1):
        idx[id+1, ] = np.roll(idx[id, ], 1)

    return idx

# ----------------------------------------------------------------------------------------------------------------------
def get_subset(patients):

    pname = {}
    for i in range(len(patients)):
        pname['subset_' + str(i)] = []

    for id, subset in enumerate(pname):
        nb_patients = len(patients[id])
        subset_dim = round(.1 * nb_patients)
        inc = 0
        for i in range(9):
            pname[subset].append(patients[id][inc:inc+subset_dim])
            inc += subset_dim
        pname[subset].append(patients[id][inc:])

    return pname

# ----------------------------------------------------------------------------------------------------------------------
def split_center(patients):

    sort_patients = []

    # --- get clin data
    clin_patients = [patient for patient in patients if 'clin' in patient]
    clin_patients.sort()

    # --- get tech data
    tech_patients = [patient for patient in patients if 'tech' in patient]
    tech_patients.sort()

    # --- get patients
    sort_patients.append(clin_patients[:1388])
    sort_patients.append(clin_patients[1388:])
    sort_patients.append(tech_patients[:100])
    sort_patients.append(tech_patients[100:200])
    sort_patients.append(tech_patients[200:300])
    sort_patients.append(tech_patients[300:400])
    sort_patients.append(tech_patients[400:])

    return sort_patients

# ----------------------------------------------------------------------------------------------------------------------
def split_data(pdata, pres, training_part, validation_part, testing_part):
    """ Store path directories of each images. data are splitted according to patient instead of patches numbers. """

    # --- check folder
    pfh.create_dir(pres)

    patients = os.listdir(pdata)
    if "backup_parameters" in patients:
        patients.remove("backup_parameters")
    if "GIF" in patients:
        patients.remove("GIF")

    patients = split_center(patients)
    patient_split = get_subset(patients)
    permutation_id = get_permutation_id(10)
    subset = ['training', 'validation', 'testing']



    for id_fol in range(10):
        pres_ = os.path.join(pres, 'fold_' + str(id_fol))
        pfh.create_dir(pres_)
        a=1
        for nsubset in subset:
            with open(os.path.join(pres_, nsubset + ".txt"), 'w') as f:
                for subset_ in patient_split:
                    if nsubset == 'training':
                        idx = permutation_id[id_fol, :8]
                        for idx_ in range(idx.shape[0]):
                            patients_ = patient_split[subset_][int(idx[idx_])]
                            for pname in patients_:
                                f.write(pname + "\n")
                    if nsubset == 'validation':
                        idx_ = permutation_id[id_fol, 8]
                        patients_ = patient_split[subset_][int(idx_)]
                        for pname in patients_:
                            f.write(pname + "\n")
                    if nsubset == 'testing':
                        idx_ = permutation_id[id_fol, 9]
                        patients_ = patient_split[subset_][int(idx_)]
                        for pname in patients_:
                            f.write(pname + "\n")
    # for key in split.keys():
    #     with open(os.path.join(pres, key + ".txt"), 'w') as f:
    #         for patient in split[key]:
    #             # ppatient = os.path.join(pdata, patient)
    #             f.write(patient + "\n")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # --- VARIABLES DECLARATION
    pdata   = '/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/SEGMENTATION/DATA'
    pres    = '/home/laine/Documents/PROJECTS_IO/CARODEEPFLOW/SEGMENTATION/SPLIT_PATIENT'
    training_part   = 0.8
    validation_part = 0.1
    testing_part    = 0.1

    # --- split data into training/validation/testing part
    split_data(pdata, pres, training_part, validation_part, testing_part)