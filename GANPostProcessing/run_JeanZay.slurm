#!/bin/sh

#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH -t 0-20:00
#SBATCH --mem=24gb                # Job memory request
#SBATCH --output=test.log	  # Standard output and error log

# --- ACTIVATE CONDA ENV
source /gpfswork/rech/obh/uby91ul/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

export MPLBACKEND='Agg' # solve pb with matplotlib

# run the script
WD=/linkhome/rech/gencre01/uby91ul/REPOSITORIES/carotid_US_DL_tool/POSTPROCESSING/GAN
cd $WD
PYTHONPATH=$WD python scripts/train.py -param $param
