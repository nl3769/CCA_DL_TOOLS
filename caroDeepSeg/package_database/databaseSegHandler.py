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
        """ Store path in dictionary to load data and patient information. """

        pname = self.parameters.PDATA.split('/')[-1]
        self.path_data['image_information'] = os.path.join(self.parameters.PDATA, 'CF-' + pname + ".txt")
        self.path_data['path_image']        = os.path.join(self.parameters.PDATA, 'images-' + pname + ".pkl")
        self.path_data['path_field']        = os.path.join(self.parameters.PDATA, 'displacement_field-' + pname + ".pkl")
        self.path_data['path_LI']           = os.path.join(self.parameters.PDATA, 'LI-' + pname + ".pkl")
        self.path_data['path_MA']           = os.path.join(self.parameters.PDATA, 'MA-' + pname + ".pkl")

    # ------------------------------------------------------------------------------------------------------------------
    def create_database(self):

        # --- we get pairs of images
        pairs = []
        seq, OF, LI, MA, CF = dbu.load_prepared_data(self.path_data)
        seq_length = seq.shape[-1]
        pname = self.parameters.PDATA.split('/')[-1]

        # --- set seq ID to save results
        for id in range(1, seq_length):

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

        for id in tqdm(range(0, (seq_length-1))):

            # --- get size of the original image
            LI1 = LI[..., id].copy()
            LI2 = LI[..., id+1].copy()
            MA1 = MA[..., id].copy()
            MA2 = MA[..., id + 1].copy()
            I1 = seq[..., id].copy()
            I2 = seq[..., id + 1].copy()
            OF_ = OF[..., id].copy()

            args_preprocessing = {
                'I1'          : I1,
                'I2'          : I2,
                'OF'          : OF_,
                'LI1'         : LI1,
                'LI2'         : LI2,
                'MA1'         : MA1,
                'MA2'         : MA2,
                'roi_width'   : self.parameters.ROI_WIDTH,
                'pixel_width' : self.parameters.PIXEL_WIDTH,
                'CF'          : CF}

            I1, I2, OF12, LI1, LI2, MA1, MA2, rCF = dbu.preprocessing_prepared_data(**args_preprocessing)
            # --- get borders
            roi_borders = dbu.get_roi_borders(LI1, LI2, MA1, MA2)
            # --- adapt segmentation to borders
            LI1, LI2, MA1, MA2 = dbu.adapt_seg_borders(LI1, LI2, MA1, MA2, roi_borders)
            # --- compute position for cropping
            mean1 = dbu.mean_pos(LI1, MA1)
            mean2 = dbu.mean_pos(LI2, MA2)
            args_coordinates = {
                "roi_borders"   : roi_borders,
                "pos1"          : mean1,
                "pos2"          : mean2,
                "shift_x"       : self.parameters.SHIFT_X,
                "shift_z"       : self.parameters.SHIFT_Z,
                "roi_width"     : self.parameters.PIXEL_WIDTH,
                "roi_height"    : self.parameters.PIXEL_HEIGHT}
            coordinates = dbu.get_cropped_coordinates(**args_coordinates)
            # --- extract data
            args_data_extraction = {
                "LI1"           : LI1,
                "LI2"           : LI2,
                "MA1"           : MA1,
                "MA2"           : MA2,
                "I1"            : I1,
                "I2"            : I2,
                "OF"            : OF12,
                "coordinates"   : coordinates,
                "pixel_width"   : self.parameters.PIXEL_WIDTH,
                "pixel_height"  : self.parameters.PIXEL_HEIGHT,
                "pairs_name"    : pairs[id]
                }
            data = dbu.data_extraction(**args_data_extraction)
            # --- save data
            if id+1 >= 100:
                substr = str(id + 1)
            elif id+1 >= 10:
                substr = "0" + str(id + 1)
            else:
                substr = "00" + str(id + 1)
            folder = os.path.join(pname, 'id_' + substr)
            dbu.save_data(data, rCF, self.parameters.PRES, folder)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_patients()
        self.create_database()

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
