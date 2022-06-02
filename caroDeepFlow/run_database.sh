#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/caroDeepFlow
PYTHONPATH=$WD python run/run_database.py -param set_parameters_database.py
