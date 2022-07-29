#!/bin/bash

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pytorch

WD=/home/laine/cluster/REPOSITORIES/CCA_DL_TOOLS/GANPostProcessing

cd $WD

PYTHONPATH=$WD python package_cores/run_train_evaluate_model.py -param lbd-GAN:1-over-30_lbd-pxl:1_kernel:3-3_loss:L1L2.py
