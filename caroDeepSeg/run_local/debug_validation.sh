#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python -m pudb package_cores/run_evaluation_segmentation_full_roi.py
