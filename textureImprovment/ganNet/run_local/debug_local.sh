#!/bin/bash

source ~/venv/pytorch/bin/activate

WD=/home/laine/Documents/REPO/CCA_DL_TOOLS/textureImprovment/ganNet

cd $WD

PYTHONPATH=$WD python -m pudb package_cores/run_train_evaluate_model.py -param GAN_parameters_template.py
