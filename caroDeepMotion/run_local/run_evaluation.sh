#!/bin/bash

source ~/venv/pytorch/bin/activate
WD=/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/caroDeepMotion/
cd $WD

PYTHONPATH=$WD python package_cores/run_evaluation.py