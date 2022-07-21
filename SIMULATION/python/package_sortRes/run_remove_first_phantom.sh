#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

wd=/home/laine/Documents/REPO/caro_deep_flow/SIMULATION/package_python

PYTHONPATH=$wd python run_remove_first_phantom.py
