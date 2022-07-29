#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/textureImprovment/diffusionNet
cd $WD
PYTHONPATH=$WD python -m pudb package_cores/run_train_evaluate_diffusionNet.py -param set_parameters_diffusionNet_template.py
