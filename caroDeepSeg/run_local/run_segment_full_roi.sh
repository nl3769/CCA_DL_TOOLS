#!/bin/bash

source ~/venv/pytorch/bin/activate
WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python package_cores/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f1.py
