#!/bin/bash

source ~/venv/pytorch/bin/activate
WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD

PYTHONPATH=$WD python package_core/run_database.py -param set_parameters_database.py