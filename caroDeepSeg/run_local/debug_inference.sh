#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python -m pubd package_cores/run_segment_full_roi.py -param set_parameters_IMC_segmentation_inference_f0.py
