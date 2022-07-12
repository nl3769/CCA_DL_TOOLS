#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/caroDeepFlow
PYTHONPATH=$WD python package_core/run_training.py -param set_parameters_training.py
