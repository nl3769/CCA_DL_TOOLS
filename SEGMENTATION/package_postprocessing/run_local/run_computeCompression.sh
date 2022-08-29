#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/SEGMENTATION/package_postprocessing
cd $WD
PYTHONPATH=$WD python package_cores/run_computeCompression.py -param set_parameters_computeCompression_segGT.py
PYTHONPATH=$WD python package_cores/run_computeCompression.py -param set_parameters_computeCompression_seg_caroSegDeep-GAN-without_fine_tuning.py
PYTHONPATH=$WD python package_cores/run_computeCompression.py -param set_parameters_computeCompression_seg_caroSegDeep-SIMULATED-without_fine_tuning.py

