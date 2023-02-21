#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepMotion
PYTHONPATH=$WD python package_core/run_database_motion.py -param set_parameters_database_motion_template.py
