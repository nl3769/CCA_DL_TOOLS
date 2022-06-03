#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

wd=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/python

PYTHONPATH=$wd python run_sort_res.py
