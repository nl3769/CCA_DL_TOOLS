#!/bin/sh

#SBATCH --time 05:00:00
#SBATCH --ntasks 1
#SBATCH --partition htc              
#SBATCH --mem 20000
#SBATCH --mail-type=END,FAIL  
#SBATCH --mail-user nolann.laine@creatis.insa-lyon.fr



cd /sps/creatis/nlaine/exec/image_reconstruction
bash run_image_reconstruction.sh /sps/creatis/nlaine/MATLAB_RUNTIME/v911 $pres $flag_cluster

# --- RUN THE SCRIPT
# cd /sps/creatis/nlaine/REPO/carotid_US_DL_tool/SIMULATION/function

# echo $pres
# echo $flag_cluster

# matlab -r 'fct_run_image_reconstruction("'$pres'","'$flag_cluster'")'
