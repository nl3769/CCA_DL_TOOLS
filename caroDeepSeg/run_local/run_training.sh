#!/bin/bash


source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python package_cores/run_training.py -param set_parameters_training.py