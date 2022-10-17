#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/SEGMENTATION/package_postprocessing
cd $WD
PYTHONPATH=$WD python -m pudb package_cores/run_gatherResults.py -param set_parameters_gatherResults_template.py
