#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/caroDeepFlow
PYTHONPATH=$WD python package_core/run_database.py -param set_parameters_database.py
