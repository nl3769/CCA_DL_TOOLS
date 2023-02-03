'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os

from package_parameters.parameters_IMC_seg_inference           import Parameters
from shutil                                                             import copyfile
import package_utils.fold_handler                                       as fh

# ----------------------------------------------------------------------------------------------------------------------
def setParameters():

  p = Parameters(
        PDATA='/home/laine/Desktop/data_cubs/images',                                                                   # PATH TO LOAD DATA
        PSEG_REF='/home/laine/Desktop/data_cubs/LIMA-Profiles-interpolated/Manual-A1',                                  # PATH TO SAVE TRAINING RESULTS
        PCF='/home/laine/Desktop/data_cubs/CF',                                                                         # PATH TO CALIBRATION FACTOR (PIXEL SIZE)
        PMODELIMC='/home/laine/Documents/PROJECTS_IO/CARODEEPSEG/TRAINING/SEG_TRAINING_FOLD_0/saved_model/netSeg_val.pth',  # PATH TO LOAD TRAINED MODEL FOR IMC SEGMENTATION
        PMODELFW=None,                                                                                                  # IF MODEL FOR FAR WALL DETECTION WAS TRAINED (NOT IMPLEMENTED YET)
        PRES='/home/laine/Documents/PROJECTS_IO/CARODEEPSEG/INFERENCE_FULL_ROI/TST',                                    # PATH TO SAVE TRAINING RESULTS
        EXPNAME='FOLD_0',                                                                                               # FOLD
        PATH_FW_REFERENCES='/home/laine/Documents/PROJECTS_IO/DATA/CUBS/FAR_WALL_DETECTION_CREATIS',                    # has to be a path if FW_INITIALIZATION='FW_PRED'
        PATIENT_NAME='/home/laine/Documents/PROJECTS_IO/CARODEEPSEG/SPLIT_PATIENT/fold_0/testing.txt',                  # PATH TO LOAD TESTING PATIENT
        DEVICE='cuda',                                                                                                  # SELECT DEVICE TO RUN (cuda or cpu)
        FW_DETECTION='MANUAL',                                                                                          # select manual, expert or automatic
        ROI_WIDTH=5e-3,                                                                                                 # SIZE OF THE ROI WIDTH
        PIXEL_WIDTH=256,                                                                                                # NUMBER OF PIXEL IN X DIRECTION OF THE SLIDING WINDOW (IT IS EQUAL TO ROI_WIDTH)
        PIXEL_HEIGHT=256,                                                                                               # NUMBER OF PIXEL IN Z DIRECTION OF THE SLIDING WINDOW
        SHIFT_X=32,                                                                                                     # X SHIFT TO GENERATE DATASET
        SHIFT_Z=64,                                                                                                     # Z SHIFT TO GENERATE DATASET
        FW_INITIALIZATION='FW_PRED',                                                                                    # chose 'FW_PRED', 'GT', 'GUI'
        # --- parameters for unet
        KERNEL_SIZE=(3, 3),                                                                                             # KERNEL SIZE OF THE MODEL
        PADDING=(1, 1),                                                                                                 # PADDING SIZE OF THE MODEL
        USE_BIAS=True,                                                                                                  # USE BIAS OR NOT
        NGF=32,                                                                                                         # NUMBER OF INPUT FEATURE OF THE UNET
        NB_LAYERS=4                                                                                                     # NUMBER OF LAYER OF THE UNET
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
