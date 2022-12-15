#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/caroDeepFlow
PYTHONPATH=$WD python -m pudb package_core/run_flyingChairs_handler.py -param set_parameters_flyingChairHandler.py
