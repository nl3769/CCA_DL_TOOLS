#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate pytorch

WD=~/cluster/REPOSITORIES/carotid_US_DL_tool/POSTPROCESSING/GAN

cd $WD

PYTHONPATH=$WD python scripts/train.py -param GAN_parameters_JZ_kernel_local.py
# PYTHONPATH=$WD python scripts/evaluation.py -param GAN_parameters.py
