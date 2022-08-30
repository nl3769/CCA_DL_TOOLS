#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/textureImprovment/ganNet

cd $WD

PYTHONPATH=$WD python package_cores/run_train_evaluate_model.py -param dilated_Unet_parameters_local_L2.py
