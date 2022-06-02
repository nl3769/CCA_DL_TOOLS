#!/bin/sh


#SBATCH --time 08:00:00
#SBATCH --ntasks 1
#SBATCH --partition htc              
#SBATCH --mem 15000
#SBATCH --mail-type=END,FAIL  
#SBATCH --mail-user nolann.laine@creatis.insa-lyon.fr


cd /sps/creatis/nlaine/exec/cluster_RF
bash run_cluster_RF.sh /sps/creatis/nlaine/MATLAB_RUNTIME/v911 $path_RF


# # --- RUN THE SCRIPT
# cd /sps/creatis/nlaine/REPO/carotid_US_DL_tool/SIMULATION/function
# #matlab -nojvm -nodisplay -r fct_run_parameters('${pfolder}','{$dname},'${pres}','${info}','${soft}','${acq_mode}')

# echo $path_res

# matlab -r 'fct_cluster_RF("'$path_RF'")'
