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

    def __init__(self, p, pimage):

        self.parameters = p
        self.pimage = pimage
        self.path_data = {}

    # ------------------------------------------------------------------------------------------------------------------
    def get_patients(self):
        """ Store path in dictionary to load data and patient information. """

        pname = self.pimage.split('/')[-1].split('.')[0]
        self.path_data['image_information'] = os.path.join(self.parameters.PDATA, 'CF', pname + '_CF' + ".txt")
        self.path_data['path_image']        = self.pimage

        files = os.listdir(os.path.join(self.parameters.PDATA, 'CONTOURS', 'A1'))
        contours = [file for file in files if pname in file]
        self.path_data['path_LI'] = os.path.join(self.parameters.PDATA, 'CONTOURS', 'A1', [seg for seg in contours if 'IFC3' in seg][0])
        self.path_data['path_MA'] = os.path.join(self.parameters.PDATA, 'CONTOURS', 'A1', [seg for seg in contours if 'IFC4' in seg][0])

    # ------------------------------------------------------------------------------------------------------------------
    def create_database(self):

        # --- load data
        I, LI, MA, CF = dbu.load_cubs_data(self.path_data)
        pname = self.pimage.split('/')[-1]

        args_preprocessing = {
            'I'          : I,
            'LI'         : LI,
            'MA'         : MA,
            'roi_width'   : self.parameters.ROI_WIDTH,
            'pixel_width' : self.parameters.PIXEL_WIDTH,
            'CF'          : CF}

        I, LI, MA, rCF = dbu.preprocessing_prepared_data_cubs(**args_preprocessing)
        # --- get borders
        roi_borders = dbu.get_roi_borders_cubs(LI, MA)
        # --- adapt segmentation to borders
        LI, MA = dbu.adapt_seg_borders_cubs(LI, MA, roi_borders)
        # --- compute position for cropping
        mean = dbu.mean_pos(LI, MA)
        args_coordinates = {
            "roi_borders"   : roi_borders,
            "pos"          : mean,
            "shift_x"       : self.parameters.SHIFT_X,
            "shift_z"       : self.parameters.SHIFT_Z,
            "roi_width"     : self.parameters.PIXEL_WIDTH,
            "roi_height"    : self.parameters.PIXEL_HEIGHT
        }
        coordinates = dbu.get_cropped_coordinates_cubs(**args_coordinates)
        # --- extract data
        args_data_extraction = {
            "LI"            : LI,
            "MA"            : MA,
            "I"             : I,
            "coordinates"   : coordinates,
            "pixel_width"   : self.parameters.PIXEL_WIDTH,
            "pixel_height"  : self.parameters.PIXEL_HEIGHT,
        }
        data = dbu.data_extraction_cubs(**args_data_extraction)
        # --- save data
        dbu.save_data_cubs(data, rCF, self.parameters.PRES, pname.split('.')[0])

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.get_patients()
        self.create_database()

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
