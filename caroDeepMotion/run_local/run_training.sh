#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepMotion
cd $WD
PYTHONPATH=$WD python package_cores/run_training_flow.py -param set_parameters_training_RAFT_template.py