"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import imageio
import package_utils.reader                     as rd
import package_database.database_utils          as dbu
import package_utils.fold_handler               as pufh

# ----------------------------------------------------------------------------------------------------------------------
class gifMaker():

    def __init__(self, p):

        self.parameters = p
        self.path_data = {}
    # ------------------------------------------------------------------------------------------------------------------

    def get_patients(self):
        """ Store path in dictionnary to load patient informations. """

        # required_keys -> patient_name/seq_id/path_image/path
        #                -> patient_name/seq_id/path_flow/path
        #                -> patient_name/seq_id/path_seg_LI/path
        #                -> patient_name/seq_id/path_seg_LA/path
        #                -> patient_name/seq_id/path_info/path

        patients_name = sorted(os.listdir(self.parameters.PDATA))

        for id, patient in enumerate(patients_name):

            self.path_data[patient] = {}
            p_res = os.path.join(self.parameters.PDATA, patient)

            # --- get path
            path_image = rd.get_fname(dir=p_res, sub_str='_bmode.png', fold=os.path.join('bmode_result', 'RF'))
            path_info = rd.get_fname(dir=p_res, sub_str='image_information', fold='phantom')
            path_LI = rd.get_fname(dir=p_res, sub_str='LI.', fold='phantom')
            path_MA = rd.get_fname(dir=p_res, sub_str='MA.', fold='phantom')
            path_parameters = rd.get_fname(dir=p_res, sub_str='parameters.mat', fold='parameters')

            if id > 0:
                path_flow = rd.get_fname(dir=p_res, sub_str='OF_', fold='phantom')
                self.path_data[patient]['path_flow'] = path_flow
            else:
                self.path_data[patient]['path_flow'] = ''

            self.path_data[patient]['image_information'] = path_info
            self.path_data[patient]['path_image'] = path_image
            self.path_data[patient]['path_LI'] = path_LI
            self.path_data[patient]['path_MA'] = path_MA
            self.path_data[patient]['path_parameters'] = path_parameters

    # ------------------------------------------------------------------------------------------------------------------
    def make_gif(self):

        # --- we get sequence of images
        seq = list(self.path_data.keys())

        # --- get path
        path = dbu.get_path_GIF(self.path_data, seq, id)

        # --- load data
        I, LI, MA, CF, seg_dim, z_start = dbu.load_data_GIF(path)

        # --- get size of the original image
        args_preprocessing = {'I': I,
                              'LI': LI,
                              'MA': MA}
        LI, MA = dbu.preprocessing_GIF(**args_preprocessing)
        I_a = dbu.add_annotation(I, LI, MA)
        pres = os.path.join(self.parameters.PRES, 'GIF')
        pufh.create_dir(pres)
        pres = os.path.join(pres, self.parameters.PDATA.split('/')[-1] + '.gif')
        imageio.mimsave(pres, I_a, fps=5)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_patients()
        # self.create_database()
        self.make_gif()

    # ------------------------------------------------------------------------------------------------------------------