import os
import random

import package_utils.fold_handler as pfh


def split_data(pdata, pres, training_part, validation_part, testing_part):
    """ Store path directories of each images. data are splitted according to patient instead of patches numbers. """

    # --- check folder
    pfh.create_dir(pres)

    patients = os.listdir(pdata)
    if "backup_parameters" in patients:
        patients.remove("backup_parameters")
    if "GIF" in patients:
        patients.remove("GIF")
    random.shuffle(patients)

    nb_patients = len(patients)
    split = {}

    nb_training = round(nb_patients * training_part)
    nb_validation = round(nb_patients * validation_part)

    split['training_patients'] = patients[:nb_training]
    split['validation_patients'] = patients[nb_training:nb_training+nb_validation]
    split['testing_patients'] = patients[nb_training+nb_validation:]

    for key in split.keys():
        with open(os.path.join(pres, key + ".txt"), 'w') as f:
            for patient in split[key]:
                f.write(patient + "\n")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # --- VARIABLES DECLARATION
    pdata = '/run/media/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/database_training_IMAGENET'
    pres = '/run/media/laine/DISK/PROJECTS_IO/MOTION/IN_SILICO/REAL_DATA/SPLIDATA'
    training_part = 0.7
    validation_part = 0.1
    testing_part = 0.2

    # --- split data into training/validation/testing part
    split_data(pdata, pres, training_part, validation_part, testing_part)
