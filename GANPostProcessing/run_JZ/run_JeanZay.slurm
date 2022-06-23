#!/bin/sh

#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=4       # CPU cores/threads
#SBATCH -t 0-20:00

# --- ACTIVATE CONDA ENV
source /gpfswork/rech/obh/uby91ul/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

export MPLBACKEND='Agg' # solve pb with matplotlib

# package_core the script
WD=/linkhome/rech/gencre01/uby91ul/REPOSITORIES/CCA_DL_TOOLS/GANPostProcessing
cd $WD
echo $param
PYTHONPATH=$WD python package_core/run_train_model.py -param $param
