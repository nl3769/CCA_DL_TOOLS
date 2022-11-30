'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os

from package_parameters.parameters_IMC_seg_inference_template           import Parameters
from shutil                                                             import copyfile
import package_utils.fold_handler                                       as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

  p = Parameters(
        PDATA='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/DATASET_for_CREATIS/images',        # PATH TO LOAD DATA
        PSEG='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/LIMA-Profiles/Manual-A1',              # PATH TO SAVE TRAINING RESULTS
        PCF='/home/laine/cluster/PROJECTS_IO/DATA/MEIBURGER/CF',
        PMODELIMC='/home/laine/cluster/PROJECTS_IO/CARODEEPSEG/TRAINING/SEG_TRAINING_FOLD_0/saved_model/netSeg_val.pth',
        PMODELFW=None,
        PRES='/home/laine/cluster/PROJECTS_IO/CARODEEPSEG/INFERENCE_FULL_ROI',                        # PATH TO SAVE TRAINING RESULTS
        EXPNAME='FOLD_0',
        PATH_FW_REFERENCES = '/home/laine/Documents/PROJECTS_IO/DATA/MEIBURGER/DATASET_for_CREATIS/LIMA-Profiles-interpolated/Manual-A1',
        PATIENT_NAME='/home/laine/cluster/PROJECTS_IO/CARODEEPSEG//SPLIT_PATIENT/fold_0/testing.txt',
        DEVICE='cuda',
        FW_DETECTION='MANUAL',                                                      # select manual, expert or automatic
        ROI_WIDTH=5e-3,                                                 # SIZE OF THE ROI WIDTH
        PIXEL_WIDTH=256,                                                # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW (IT IS EQUAL TO ROI_WIDTH)
        PIXEL_HEIGHT=256,                                               # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW
        SHIFT_X=32,                                                     # X SHIFT TO GENERATE DATASET
        SHIFT_Z=64,                                                     # Z SHIFT TO GENERATE DATASET
        INITIALIZATION_FROM_REFERENCES=True,
        # --- parameters for unet
        KERNEL_SIZE=(3, 3),
        PADDING=(1, 1),
        USE_BIAS=True,
        NGF=32,                 # number of input features of the Unet
        NB_LAYERS=4
  )

  pparam = os.path.join(p.PRES, p.EXPNAME, 'parameters')
  fh.create_dir(p.PRES)
  fh.create_dir(pparam)

  # --- Print all attributes in the console
  attrs = vars(p)
  print('\n'.join("%s: %s" % item for item in attrs.items()))
  print('----------------------------------------------------------------')

  # --- Save a backup of the parameters so it can be tracked on Git, without requiring to be adapted by from other contributors
  copyfile(os.path.join('package_parameters', os.path.basename(__file__)), os.path.join(pparam, 'get_parameters_training.py'))

  # --- Modify the function name from "setParameters" to "getParameters"
  fid = open(os.path.join(pparam, 'get_parameters_training.py'), 'rt')
  data = fid.read()
  data = data.replace('setParameters()', 'getParameters()')
  fid.close()
  fid = open(os.path.join(pparam, 'get_parameters_training.py'), 'wt')
  fid.write(data)
  fid.close()

  # --- Return populated object from Parameters class
  return p
