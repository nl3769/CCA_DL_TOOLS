#!/bin/bash


source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python package_cores/run_evaluation_segmentation_full_roi.py -param set_parameters_evaluation_template.py