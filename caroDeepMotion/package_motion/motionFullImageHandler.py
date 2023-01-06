"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import torch
import matplotlib.pyplot                        as plt
import package_motion.utils                     as pmutl
import package_database.database_utils          as dbu
import numpy                                    as np

from tqdm                                       import tqdm
from scipy.ndimage                              import gaussian_filter
import package_utils.fold_handler               as pufh
import package_utils.saver                      as pus

# ----------------------------------------------------------------------------------------------------------------------
class motionFullImgHandler():

    def __init__(self, p, netEncoder, netFlow, device):

        self.parameters = p
        self.path_data = {}
        self.netEncoder = netEncoder
        self.netFlow = netFlow
        self.device = device

    # ------------------------------------------------------------------------------------------------------------------
    def get_patients(self):
        """ Store path in dictionary to load data and patient information. """

        pname = self.parameters.PDATA.split('/')[-1]

        self.path_data['image_information'] = os.path.join(self.parameters.PDATA, 'CF-' + pname + ".txt")
        self.path_data['path_image'] = os.path.join(self.parameters.PDATA, 'images-' + pname + ".pkl")
        self.path_data['path_field'] = os.path.join(self.parameters.PDATA, 'displacement_field-' + pname + ".pkl")
        self.path_data['path_LI'] = os.path.join(self.parameters.PDATA, 'LI-' + pname + ".pkl")
        self.path_data['path_MA'] = os.path.join(self.parameters.PDATA, 'MA-' + pname + ".pkl")

    # ------------------------------------------------------------------------------------------------------------------
    def get_data(self):

        # --- we get pairs of images
        pairs = []
        seq, OF, LI, MA, CF = dbu.load_prepared_data(self.path_data)
        [height_seq, width_seq, nb_frame] = seq.shape
        pname = self.parameters.PDATA.split('/')[-1]

        # --- set seq ID to save results
        for id in range(1, nb_frame):
            if id < 10:
                p0 = pname + "_00" + str(id)
            elif id < 100:
                p0 = pname + "_0" + str(id)
            else:
                p0 = pname + "_" + str(id)
            if id + 1 < 10:
                p1 = pname + "_00" + str(id+1)
            elif id + 1 < 100:
                p1 = pname + "_0" + str(id + 1)
            else:
                p1 = pname + "_" + str(id + 1)
            pairs.append([p0, p1])

        for id in range(0, (nb_frame-1)):
            # --- get size of the original image
            I1 = seq[..., id].copy()
            I2 = seq[..., id + 1].copy()
            OF_ = OF[..., id].copy()
            args_preprocessing = {
                'I1'          : I1,
                'I2'          : I2,
                'OF'          : OF_,
                'roi_width'   : self.parameters.ROI_WIDTH,
                'pixel_width' : self.parameters.PIXEL_WIDTH,
                'CF'          : CF}
            I1, I2, OF12, rCF = pmutl.scale_images(**args_preprocessing)
            args_coordinates = {
                "shift_x"       : self.parameters.SHIFT_X,
                "shift_z"       : self.parameters.SHIFT_Z,
                "roi_width"     : self.parameters.PIXEL_WIDTH,
                "roi_height"    : self.parameters.PIXEL_HEIGHT,
                "dim_img"       : I1.shape}
            coordinates = pmutl.get_coordinates_full_img(**args_coordinates)
            # --- extract data
            args_data_extraction = {
                "I1"            : I1,
                "I2"            : I2,
                "OF"            : OF12,
                "coordinates"   : coordinates,
                "pixel_width"   : self.parameters.PIXEL_WIDTH,
                "pixel_height"  : self.parameters.PIXEL_HEIGHT,
                "pairs_name"    : pairs[id]}
            data = pmutl.patches_extraction(**args_data_extraction)

            data['dim_img'] = I1.shape
            data['patch_dim'] = (256, 256)
            data['I1_org'] = I1.copy()
            data['I2_org'] = I2.copy()
            data['OF_org'] = OF12.copy()
            data['CF'] = rCF

            return data

    # ------------------------------------------------------------------------------------------------------------------
    def data_preprocessing(self, data):
        # --- I1
        for id in range(len(data['I1'])):
            I_ = data['I1'][id].copy()
            I_ -= np.min(I_)
            I_ /= np.max(I_)
            data['I1'][id] = I_.copy()

        for id in range(len(data['I2'])):
            I_ = data['I2'][id].copy()
            I_ -= np.min(I_)
            I_ /= np.max(I_)
            data['I2'][id] = I_.copy()

        return data
    # ------------------------------------------------------------------------------------------------------------------
    def model_inference(self, data):

        I1_patches = data['I1']
        I2_patches = data['I2']
        batch_size = 4
        nb_batch = int(np.ceil(len(I1_patches)/batch_size))
        flow_pred = []

        for id_batch in tqdm(range(nb_batch)):
            if id_batch < nb_batch-1:
                I1 = I1_patches[id_batch * batch_size: (id_batch + 1) * batch_size]
                I2 = I2_patches[id_batch * batch_size: (id_batch + 1) * batch_size]
            else:
                I1 = I1_patches[id_batch * batch_size:]
                I2 = I2_patches[id_batch * batch_size:]

            I1 = torch.from_numpy(np.expand_dims(np.array(I1), axis=1)).to(self.device)
            I2 = torch.from_numpy(np.expand_dims(np.array(I2), axis=1)).to(self.device)
            mask = torch.ones(I1.shape).to(self.device)

            fmap1, skc1, fmap2, skc2 = self.netEncoder(I1, I2)
            pred = self.netFlow(I1, fmap1, fmap2, mask)
            pred = np.array(pred[-1].detach().to('cpu'))

            for id_pred in range(pred.shape[0]):
                flow_pred.append(pred[0, ])

        data['flow_pred'] = flow_pred

        return data

    # ------------------------------------------------------------------------------------------------------------------
    def mosaic(self, data):

        flow_pred_overlap = np.zeros((2,) + data['dim_img'])
        superimposed_map = np.zeros((2,) + data['dim_img'])
        pixel_height = data['patch_dim'][0]
        pixel_width = data['patch_dim'][1]
        ones_map = np.ones((2,) + data['patch_dim'])
        for id_patch in range(len(data['coord'])):
            coord = data['coord'][id_patch]
            x, z = coord[0], coord[1]
            pred = data['flow_pred'][id_patch].copy()
            # pred = np.transpose(data['OF'][id_patch][...,0::2].copy(), (2,0,1))
            flow_pred_overlap[:, z:z+pixel_height, x:x+pixel_width] += pred
            superimposed_map[:, z:z + pixel_height, x:x + pixel_width] += ones_map

        flow_pred_overlap = flow_pred_overlap / superimposed_map
        flow_pred_overlap[0, ] = gaussian_filter(flow_pred_overlap[0, ].copy(), 20)
        flow_pred_overlap[1, ] = gaussian_filter(flow_pred_overlap[1, ].copy(), 20)

        data['flow_pred_final'] = flow_pred_overlap

    # ------------------------------------------------------------------------------------------------------------------
    def save_res(self, pres, data):

        pufh.create_dir(pres)

        # --- save flow org
        flow_gt = np.transpose(data['OF_org'][:, :, 0::2], (2, 0, 1))
        pres_ = os.path.join(pres, 'flow_gt.pkl')
        pus.write_pickle(flow_gt, pres_)

        # --- save flow org
        flow_pred = data['flow_pred_final']
        pres_ = os.path.join(pres, 'flow_pred.pkl')
        pus.write_pickle(flow_pred, pres_)

        # --- save images
        I1 = data['I1_org']
        I2 = data['I2_org']
        I = np.zeros((2,) + I1.shape)
        I[0, ] = I1
        I[1, ] = I2
        pres_ = os.path.join(pres, 'images.pkl')
        pus.write_pickle(I, pres_)

        # --- save cf
        pres_ = os.path.join(pres, 'cf.txt')
        with open(pres_, 'w') as f:
            for key, value in data['CF'].items():
                f.write('%s:%s\n' % (key, value))

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_patients()
        data = self.get_data()
        self.data_preprocessing(data)
        self.model_inference(data)
        self.mosaic(data)
        self.save_res(self.parameters.PSAVE, data)
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------