#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/textureImprovment/ganNet

cd $WD

PYTHONPATH=$WD python -m pudb package_cores/run_train_evaluate_model.py -param dilated_Unet_parameters_local_L2.py
