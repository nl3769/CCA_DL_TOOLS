#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepMotion
PYTHONPATH=$WD python -m pudb package_core/run_flyingChairs_handler.py -param set_parameters_flyingChairHandler.py
