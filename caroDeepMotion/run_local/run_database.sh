#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/caroDeepFlow
PYTHONPATH=$WD python package_core/run_database.py -param set_parameters_database.py
