#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepMotion
PYTHONPATH=$WD python package_core/run_database_preparation.py -param set_parameters_database_preparation_template.py
