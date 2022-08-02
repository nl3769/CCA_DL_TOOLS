#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/textureImprovment/ganNet

cd $WD

PYTHONPATH=$WD python -m pudb package_cores/run_train_evaluate_model.py -param GAN_parameters_template.py
