'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import shutil
import sys

# ----------------------------------------------------------------------------------------------------------------------
class dataSetBuilder:

    def __init__(self, info):

        self.info = info
        self.names = {} # store fname

    def get_fname(self):

        # --- get file names
        fnames = os.listdir(self.info['pdata'])
        self.names = [key for key in fnames if 'id' not in key]

    # ------------------------------------------------------------------------------------------------------------------
    def extract_images(self):

        for name in self.names:
            pname = os.listdir(os.path.join(self.info['pdata'], name))
            if len(pname) != 1:
                sys.exit('error in extract_images function')
            pimage = os.path.join(self.info['pdata'], name, pname[0], 'bmode_result/RF')
            # pimage = os.path.join(self.info['pdata'], name, 'bmode_result/RF')
            files = os.listdir(pimage)
            sim = [key for key in files if 'bmode.png' in key][0]
            org = [key for key in files if 'in_vivo.png' in key][0]

            pres = os.path.join(self.info['pres'], name)
            create_directory(pres)

            src = os.path.join(pimage, org)
            dst = os.path.join(pres, name + '_org.png')
            copy_paste(src, dst)

            src = os.path.join(pimage, sim)
            dst = os.path.join(pres, name + '_simulated.png')
            copy_paste(src, dst)

    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def sort_images( fnames):
    ''' Sort file names. return dictionnary containing the name of the original image and all simulated fname of the sequence. '''
    pname = {}

    for id in range(0, len(fnames)-4, 4):

        pname[fnames[id]] = [fnames[id+1], fnames[id+2], fnames[id+3]]

    return pname

# ----------------------------------------------------------------------------------------------------------------------
def create_directory(path: str):
    ''' Create directory. '''

    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory if it does not exist
        os.makedirs(path)

# ----------------------------------------------------------------------------------------------------------------------
def copy_paste(src: str, dest: str):

    shutil.copyfile(src, dest)