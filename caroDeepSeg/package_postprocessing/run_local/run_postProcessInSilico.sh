#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/SEGMENTATION//package_postprocessing
cd $WD
PYTHONPATH=$WD python package_cores/run_postProcessInSilico.py -param set_parameters_postProcessInSilico_template.py
