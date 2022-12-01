import os
import shutil

# ----------------------------------------------------------------------------------------------------------------------
def remove_RF_folder(folder, sub_folders, rm_folder):

    patients = os.listdir(folder)

    for patient in patients:
        path_ = os.path.join(folder, patient)
        nseq = os.listdir(path_)

        for id_seq in nseq:
            pseq = os.path.join(path_, id_seq, sub_folders)
            folders = os.listdir(pseq)
            if rm_folder in folders:
                shutil.rmtree(os.path.join(pseq, rm_folder))

# ----------------------------------------------------------------------------------------------------------------------
def remove_RF_file(folder, sub_folders, fname):

    patients = os.listdir(folder)

    for patient in patients:
        path_ = os.path.join(folder, patient)
        nseq = os.listdir(path_)

        for id_seq in nseq:
            pseq = os.path.join(path_, id_seq, sub_folders)
            folders = os.listdir(pseq)
            if fname in folders:
                os.remove(os.path.join(pseq, fname))
# ----------------------------------------------------------------------------------------------------------------------
def main():

    folder = '/home/laine/HDD/PROJECTS_IO/SIMULATION/IMAGENET_STA'
    sub_folders = 'bmode_result'
    rm_folder = 'RF'
    remove_RF_folder(folder, sub_folders, rm_folder)

    fname = 'RF_raw_data.mat'
    sub_folders = 'raw_data'
    remove_RF_file(folder, sub_folders, fname)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()