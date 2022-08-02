#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/caroDeepFlow
cd $WD
PYTHONPATH=$WD python -m pudb package_cores/run_training.py -param set_parameters_training.py
