#!/bin/bash

source ~/venv/pytorch/bin/activate
WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/caroDeepSeg
cd $WD
PYTHONPATH=$WD python package_cores/run_split_data.py
