"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import argparse
import importlib
import time
import torch
import numpy                            as np
import matplotlib.pyplot                as plt
from package_inference.sequenceHandler  import sequenceClassIMC
from tqdm                               import tqdm

# ----------------------------------------------------------------------------------------------------------------------
def get_patient_name(patients_name, patient_list):

    with open(os.path.join(patients_name), 'r') as f:
        lines = f.readlines()

    lines = [key.split('.')[0].replace('\n', '') + '.tiff' for key in lines]

    intersection_set = set.intersection(set(patient_list), set(lines))
    return list(intersection_set)

# ----------------------------------------------------------------------------------------------------------------------
def save_seg(path, seq, patient):
    """ Save segmentation results in .txt format. """

    LI_ = open(os.path.join(path, patient.split('.')[0] + "-LI.txt"), "w+")
    MA_ = open(os.path.join(path, patient.split('.')[0] + "-MA.txt"), "w+")
    for k in range(seq.annotationClass.borders_org['leftBorder'], seq.annotationClass.borders_org['rightBorder'], 1):
        LI_.write(str(k) + " " + str(seq.annotationClass.map_annotation_org[0, k, 0]) + "\n")
        MA_.write(str(k) + " " + str(seq.annotationClass.map_annotation_org[0, k, 1]) + "\n")

    LI_.close()
    MA_.close()

# ----------------------------------------------------------------------------------------------------------------------
def save_image(path, seq, patient):
    """ Saves image for visual inspection. """

    img = np.zeros(seq.firstFrame.shape + (3,))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = seq.firstFrame, seq.firstFrame, seq.firstFrame

    for k in range(seq.annotationClass.borders_org['leftBorder'], seq.annotationClass.borders_org['rightBorder'] + 1, 1):
        LI_val = round(seq.annotationClass.map_annotation_org[0, k, 0])
        MA_val = round(seq.annotationClass.map_annotation_org[0, k, 1])

        img[LI_val, k, 2] = 150
        img[LI_val, k, 0] = 0
        img[LI_val, k, 1] = 0
        img[MA_val, k, 0] = 150
        img[MA_val, k, 1] = 0
        img[MA_val, k, 2] = 0

    plt.imsave(os.path.join(path, patient.split('.')[0] + ".png"), img.astype(np.uint8))

# ----------------------------------------------------------------------------------------------------------------------
def main():
    # --- use a parser with set_parameters.py
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True,
                           help='List of package_parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()
    # --- get image name
    patient_name_list = os.listdir(p.PDATA)
    patient_name_list = get_patient_name(p.PATIENT_NAME, patient_name_list)
    patient_name_list.sort()
    # --- select device
    if p.DEVICE == 'cuda':
        p.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- to write execution time
    exec_time = open(os.path.join(p.PATH_EXEC_TIME, "exec_time.txt"), "w")
    # --- to write number of patches by patient
    nb_patches = open(os.path.join(p.PATH_NB_PATCHES, "nb_patches.txt"), "w")
    # --- launch process
    for id_patient in tqdm(range(len(patient_name_list))):
        # --- create the object sequenceClass
        patientName = patient_name_list[id_patient]
        seq = sequenceClassIMC(
            path_seq=os.path.join(p.PDATA, patientName),
            p=p)
        # --- launch the segmentation
        t = time.time()
        inference_time = seq.sliding_window_vertical_scan()
        elapsed = time.time() - t
        # --- save execution timer and number of patches
        exec_time.write(str(elapsed) + "\n")
        nb_patches.write(str(len(seq.predictionClass.patches)) + "\n")
        # --- save segmentation results
        save_seg(p.PATH_SEGMENTATION_RESULTS, seq, patientName)
        # --- save image with LI/MA segmentation
        save_image(p.PATH_SEG_VISUAL, seq, patientName)
    # --- close writter
    exec_time.close()
    nb_patches.close()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------------------------------------------------
