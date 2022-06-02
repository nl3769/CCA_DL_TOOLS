#!/bin/sh


#SBATCH --time 04:00:00
#SBATCH --ntasks 1
#SBATCH --partition htc
#SBATCH --mem 15000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user nolann.laine@creatis.insa-lyon.fr

cd /sps/creatis/nlaine/exec/make_phantom 
bash run_make_phantom.sh /sps/creatis/nlaine/MATLAB_RUNTIME/v911 $pfolder $dname $pres $info $soft $acq_mode $nb_img


# # --- RUN THE SCRIPT
# cd /sps/creatis/nlaine/REPO/carotid_US_DL_tool/SIMULATION/function

# echo $pfolder
# echo $dname
# echo $pres
# echo $info
# echo $soft
# echo $acq_mode
# echo $nb_img

# ccenv matlab R2020a

# process="matlab -r fct_run_parameters('"$pfolder"','"$dname"','"$pres"','"$info"','"$soft"','"$acq_mode"','$nb_img')"

# echo $process
# $process