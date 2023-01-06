"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os

import numpy as np

import package_utils.reader                     as rd
import package_database.database_utils          as dbu
import package_debug.visualisation              as dv
import matplotlib.pyplot                        as plt
from tqdm                                       import tqdm
from icecream                                   import ic

# ----------------------------------------------------------------------------------------------------------------------
class databasePreparation():

    def __init__(self, p):

        self.parameters = p
        self.path_data = {}
    # ------------------------------------------------------------------------------------------------------------------

    def get_patients(self):
        """ Store path in dictionary to load patient information. """

        frame_id = sorted(os.listdir(self.parameters.PDATA))
        for id, nframe in enumerate(frame_id):
            self.path_data[nframe] = {} # dictionary to store path
            p_res = os.path.join(self.parameters.PDATA, nframe) # path to data
            # --- get path
            path_image = rd.get_fname(dir=p_res, sub_str='_bmode.png', fold=os.path.join('bmode_result', 'results'))
            path_info = rd.get_fname(dir=p_res, sub_str='image_information', fold='phantom')
            path_LI = rd.get_fname(dir=p_res, sub_str='LI.', fold='phantom')
            path_MA = rd.get_fname(dir=p_res, sub_str='MA.', fold='phantom')
            path_parameters = rd.get_fname(dir=p_res, sub_str=['parameters.mat', 'parameters.json'], fold='parameters')
            if id > 0:
                path_flow = rd.get_fname(dir=p_res, sub_str='OF_', fold='phantom')
                self.path_data[nframe]['path_flow'] = path_flow
            else:
                self.path_data[nframe]['path_flow'] = ''
            self.path_data[nframe]['image_information'] = path_info
            self.path_data[nframe]['path_image'] = path_image
            self.path_data[nframe]['path_LI'] = path_LI
            self.path_data[nframe]['path_MA'] = path_MA
            self.path_data[nframe]['path_parameters'] = path_parameters

    # ------------------------------------------------------------------------------------------------------------------
    def create_database(self):
        # --- we get pairs of images
        pairs = []
        keys = list(self.path_data.keys())
        for id in range(len(keys)-1):
            p0 = keys[id]
            p1 = keys[id+1]
            pairs.append([p0, p1])
        for id in range(0, len(pairs)):
            # --- get path
            path = dbu.get_path(self.path_data, pairs, id)
            # --- load data
            I1, I2, OF, LI1, LI2, MA1, MA2, CF, seg_dim, z_start = dbu.load_data(path)
            # --- get size of the original image
            args_preprocessing = {
                'I1'          : I1,
                'I2'          : I2,
                'OF'          : OF,
                'LI1'         : LI1,
                'LI2'         : LI2,
                'MA1'         : MA1,
                'MA2'         : MA2,
                'pairs'       : pairs[id],
                'CF'          : CF,
                'zstart'      : z_start}
            I1, I2, OF, LI1, LI2, MA1, MA2 = dbu.data_preparation_preprocessing(**args_preprocessing)
            if id == 0:
                I_seq = np.zeros(I1.shape + (len(pairs) + 1,))
                OF_seq = np.zeros(OF.shape + (len(pairs),))
                LI_seq = np.zeros(LI1.shape + (len(pairs) + 1,))
                MA_seq = np.zeros(MA1.shape + (len(pairs) + 1,))
                I_seq[..., 0], I_seq[..., 1] = I1, I2
                OF_seq[..., 0] = OF
                LI_seq[:, 0], MA_seq[:, 0] = LI1, MA1
                LI_seq[:, 1], MA_seq[:, 1] = LI2, MA2
            else:
                I_seq[..., id+1] = I2
                OF_seq[..., id] = OF
                LI_seq[:, id+1], MA_seq[:, id+1] = LI2, MA2
        dbu.save_data_preparation(I_seq, OF_seq, LI_seq, MA_seq, CF, self.parameters.PRES, self.parameters.PDATA.split('/')[-1])
        dbu.mk_animation(self.parameters.PRES, self.parameters.PDATA.split('/')[-1], CF) # can be commented on, just for visual inspection

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_patients()
        self.create_database()

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
