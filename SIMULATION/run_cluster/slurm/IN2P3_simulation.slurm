#!/bin/sh


#SBATCH --time 03:00:00
#SBATCH --ntasks 1
#SBATCH --partition htc              
#SBATCH --mem 20000
#SBATCH --mail-type=END,FAIL  
#SBATCH --mail-user nolann.laine@creatis.insa-lyon.fr
#SBATCH --array=0-192

cd /sps/creatis/nlaine/exec/wave_propagation
bash run_wave_propagation.sh /sps/creatis/nlaine/MATLAB_RUNTIME/v911 $path_param $path_phantom $flag_cluster $SLURM_ARRAY_TASK_ID


# # --- RUN THE SCRIPT
# cd /sps/creatis/nlaine/REPO/carotid_US_DL_tool/SIMULATION/function

# echo $path_param
# echo $path_phantom
# echo $flag_cluste
# ccenv matlab R2020a
# process="matlab -r fct_run_wave_propagation('$path_param','$path_phantom',$flag_cluster,$SLURM_ARRAY_TASK_ID)"
# echo $process
# $process
