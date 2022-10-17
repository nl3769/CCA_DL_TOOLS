#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/SEGMENTATION/package_postprocessing
cd $WD
PYTHONPATH=$WD python package_cores/run_gatherResults.py -param set_parameters_gatherResults_template.py
