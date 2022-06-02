"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import package_utils.reader                     as rd
import package_database.database_utils          as dbu
import package_debug.visualisation              as dv
import matplotlib.pyplot                        as plt
from tqdm                                       import tqdm
from icecream                                   import ic

# ----------------------------------------------------------------------------------------------------------------------
class databaseHandler():

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
    def create_database(self):

        # --- we get pairs of images
        pairs = []
        keys = list(self.path_data.keys())
        for id in range(len(keys)-1):
            p0 = keys[id]
            p1 = keys[id+1]
            pairs.append([p0, p1])

        for id in tqdm(range(0, len(pairs))):
            ic(id)
            # --- get path
            path = dbu.get_path(self.path_data, pairs, id)
            # --- load data
            I1, I2, OF, LI1, LI2, MA1, MA2, CF, seg_dim, z_start = dbu.load_data(path)
            plt.figure()
            plt.plot(LI1)
            # --- get size of the original image
            # args_preprocessing = {'I1': I1,
            #                       'I2': I2,
            #                       'OF': OF,
            #                       'LI1': LI1,
            #                       'LI2': LI2,
            #                       'MA1': MA1,
            #                       'MA2': MA2,
            #                       'pairs': pairs[id],
            #                       'roi_width': self.parameters.ROI_WIDTH,
            #                       'pixel_width': self.parameters.PIXEL_WIDTH,
            #                       'CF': CF,
            #                       'zstart': z_start}
            # I1, I2, OF, LI1, LI2, MA1, MA2 = dbu.preprocessing(**args_preprocessing)
            # # --- get borders
            # roi_borders = dbu.get_roi_borders(LI1, LI2, MA1, MA2)
            # # --- adapt segmentation to borders
            # LI1, LI2, MA1, MA2 = dbu.adapt_seg_borders(LI1, LI2, MA1, MA2, roi_borders)
            # # --- compute position for cropping
            # mean1 = dbu.mean_pos(LI1, MA1)
            # mean2 = dbu.mean_pos(LI2, MA2)
            # args_coordinates = {"roi_borders": roi_borders,
            #                     "pos1": mean1,
            #                     "pos2": mean2,
            #                     "shift_x": self.parameters.SHIFT_X,
            #                     "shift_z": self.parameters.SHIFT_Z,
            #                     "roi_width": self.parameters.PIXEL_WIDTH,
            #                     "roi_height": self.parameters.PIXEL_HEIGHT}
            # coordinates = dbu.get_cropped_coordinates(**args_coordinates)
            # # --- extract data
            # args_data_extraction = {"LI1": LI1,
            #                         "LI2": LI2,
            #                         "MA1": MA1,
            #                         "MA2": MA2,
            #                         "I1": I1,
            #                         "I2": I2,
            #                         "OF": OF,
            #                         "coordinates": coordinates,
            #                         "pixel_width": self.parameters.PIXEL_WIDTH,
            #                         "pixel_height": self.parameters.PIXEL_HEIGHT}
            # data = dbu.data_extraction(**args_data_extraction)
            # # --- save data
            # if id+1 >= 100:
            #     substr = str(id + 1)
            # elif id+1 >= 10:
            #     substr = "0" + str(id + 1)
            # else:
            #     substr = "00" + str(id + 1)
            # folder = os.path.join(pairs[id][1].split('id')[0][:-1], 'id_' + substr)
            # dbu.save_data(data, CF, self.parameters.PRES, folder)
        plt.show()
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_patients()
        self.create_database()

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
