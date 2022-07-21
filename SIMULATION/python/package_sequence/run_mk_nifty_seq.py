from icecream                       import ic
from package_utils.data_loader      import get_seg, get_borders
from package_utils.data_handler     import adapt_segmentation, add_annotation
from mat4py                         import loadmat
import numpy                        as np
import imageio                      as iio
import nibabel                      as nib
import os
import glob
import shutil

#  ----------------------------------------------------------------------------------------------------------------------
class sequenceHandler():

    def __init__(self, pres, pdata, substr, patient_name):

        self.pres           = pres
        self.pdata          = pdata
        self.patient_name   = patient_name
        self.substr         = substr
        self.sequence       = []
        self.nsequence      = {'pdata': [], 'nout': []}

    # -------------------------------------------------------------------------------------------------------------------
    def get_image_info(self):

        file = sorted(glob.glob(os.path.join(self.pdata, "*_id_*")))[0]
        path = os.path.join(self.pdata, file, "phantom", "image_information.mat")
        i_info = loadmat(path)
        CF = i_info['image']['CF']
        dim_org = np.array(i_info['image']['image']).shape

        return dim_org, CF

    # -------------------------------------------------------------------------------------------------------------------
    def get_simulated_image(self, pdata):

        path = pdata + self.substr
        # ic(path)
        pimg = sorted(glob.glob(os.path.join(path, '*_bmode.png')))[0]

        return pimg

    # -------------------------------------------------------------------------------------------------------------------
    def get_sequence_name(self):
        """ Sort name in correct order. """

        fnames = sorted(glob.glob(os.path.join(self.pdata, "*_id_*")))
        fullnames = list(map(self.get_simulated_image, fnames))

        self.nsequence['pdata'] = [key for key in fullnames]
        self.nsequence['nout'] = [key + '.png' for key in fnames]

    # -------------------------------------------------------------------------------------------------------------------
    def copy_data(self):

        for count, img in enumerate(self.nsequence['nout']):
            scr = self.nsequence['pdata'][count]
            dst = os.path.join(self.pres, img)
            shutil.copy(scr, dst)

    # -------------------------------------------------------------------------------------------------------------------
    def get_sequence(self):

        for img in self.nsequence['pdata']:
            # ic(img)
            self.sequence.append(iio.imread(img))

    # -------------------------------------------------------------------------------------------------------------------
    def create_gif(self):

        pres = os.path.join(self.pres, "GIF")
        check_dir(pres)
        pres = os.path.join(pres, 'animation.gif')
        iio.mimsave(pres, self.sequence, fps=10)

    # -------------------------------------------------------------------------------------------------------------------
    def get_seg(self):

        LI, MA = get_seg(self.pdata, self.patient_name)
        borders = get_borders(LI, MA)
        return LI, MA, borders

    # -------------------------------------------------------------------------------------------------------------------
    def save_seq(self, CF):
        pres = os.path.join(self.pres, "sequence")
        check_dir(pres)
        seq_array = np.zeros((len(self.sequence),) + self.sequence[0].shape)
        for i in range(len(self.sequence)):
            # ic(i)
            seq_array[i,] = self.sequence[i]

        affine = np.eye(4)
        affine[0, 0] = CF
        affine[1, 1] = CF
        ni_seq = nib.Nifti1Image(seq_array, affine)
        # print(ni_seq.header)
        nib.save(ni_seq, os.path.join(pres, self.patient_name + ".nii"))

    # -------------------------------------------------------------------------------------------------------------------
    def save_seg(self, LI, MA, borders):

        pres = os.path.join(self.pres, "segmentation")
        check_dir(pres)
        patients = LI.keys()

        for patient in patients:
            LI_ = LI[patient]
            MA_ = MA[patient]

            len_LI = len(LI_)
            len_MA = len(MA_)

            # --- save LI
            with open(os.path.join(pres, patient + '-LI.txt'), 'w') as f:
                for i in range(len_LI):
                    f.write(str(i) + ' ' + str(LI_[i][0]) + '\n')
            # --- save MA
            with open(os.path.join(pres, patient + '-MA.txt'), 'w') as f:
                for i in range(len_MA):
                    f.write(str(i) + ' ' + str(MA_[i][0]) + '\n')

    # -------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        dim_org, CF = self.get_image_info()
        self.get_sequence_name()
        # self.copy_data()
        self.get_sequence()
        LI, MA, borders = self.get_seg()
        self.save_seq(CF)
        LI, MA = adapt_segmentation(self.sequence[0].shape, dim_org, LI, MA)
        self.sequence = add_annotation(LI, MA, self.sequence)
        self.save_seg(LI, MA, borders)
        self.create_gif()

        return LI, MA, CF

# -----------------------------------------------------------------------------------------------------------------------
def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------------------------------------------------------------------------------------------------------
def main():

    patient_name    = "tech_001"
    pdata           = os.path.join('/home/laine/HDD/PROJECTS_IO/SIMULATION/SEQ_IN_SILICO', patient_name)
    pres            = os.path.join('/home/laine/Desktop/SILICO-SEQUENCES', patient_name)
    substr          = '/bmode_result/RF'
    check_dir(pres)
    sequence = sequenceHandler(pres, pdata, substr, patient_name)
    sequence()

# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
