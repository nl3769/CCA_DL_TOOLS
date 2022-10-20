#!/bin/sh

#PBS -l walltime=400:00:00
#PBS -l nodes=1:ppn=4:gpus=1 -q gpu
#PBS -l mem=24GB
#PBS -m ae
#PBS -e /home/laine/PROJECTS_IO/CARODEEPFLOW/log/$log_name.err
#PBS -o /home/laine/PROJECTS_IO/CARODEEPFLOW/log/$log_name.out
#PBS -N $log_name
#PBS -M laine@creatis.insa-lyon.fr

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# run script
WD=/home/laine/REPOSITORIES/CCA_DL_TOOLS/caroDeepMotion
cd $WD
PYTHONPATH=$WD python package_cores/run_training_flow.py -param $param
