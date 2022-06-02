#!/bin/bash

parameters=(
	    "GAN_UPCONV_KS_5_BS_4_LR_00001.py"
	    "GAN_UPCONV_KS_5_BS_4_LR_0001.py"
	    "GAN_UPCONV_KS_5_BS_4_LR_001.py"
	    "GAN_UPCONV_KS_7_BS_4_LR_00001.py"
	    "GAN_UPCONV_KS_7_BS_4_LR_0001.py"
	    "GAN_UPCONV_KS_7_BS_4_LR_001.py"
	    "GAN_UPCONV_KS_9_BS_4_LR_00001.py"
	    "GAN_UPCONV_KS_9_BS_4_LR_0001.py"
	    "GAN_UPCONV_KS_9_BS_4_LR_001.py"
	    "GAN_UPCONV_KS_7_BS_8_LR_0001.py"
	    "GAN_UPCONV_KS_9_BS_8_LR_0001.py"
	    "GAN_UPCONV_KS_5_BS_8_LR_0001.py"
	    "GAN_UPCONV_KS_5_BS_16_LR_0001.py"
	    "GAN_UPCONV_KS_5_BS_8_LR_0001_LP_30_LG_1"
	    "GAN_UPCONV_KS_5_BS_8_LR_0001_LP_20_LG_1"
)

experience=(
             "GAN_UPCONV_KS_5_BS_4_LR_00001"
             "GAN_UPCONV_KS_5_BS_4_LR_0001"
             "GAN_UPCONV_KS_5_BS_4_LR_001"
             "GAN_UPCONV_KS_7_BS_4_LR_00001"
             "GAN_UPCONV_KS_7_BS_4_LR_0001"
             "GAN_UPCONV_KS_7_BS_4_LR_001"
             "GAN_UPCONV_KS_9_BS_4_LR_00001"
             "GAN_UPCONV_KS_9_BS_4_LR_0001"
             "GAN_UPCONV_KS_9_BS_4_LR_001"
             "GAN_UPCONV_KS_7_BS_8_LR_0001"
             "GAN_UPCONV_KS_9_BS_8_LR_0001"
             "GAN_UPCONV_KS_5_BS_8_LR_0001"
             "GAN_UPCONV_KS_5_BS_16_LR_0001"
	     "GAN_UPCONV_KS_5_BS_8_LR_0001_LP_30_LG_1"
	     "GAN_UPCONV_KS_5_BS_8_LR_0001_LP_20_LG_1"
	    )

for idx in ${!parameters[@]}; do

  sbatch --export=ALL,param=${parameters[$idx]} -o ${experience[$idx]}.log -J ${experience[$idx]} run_JeanZay.slurm
  echo ${parameters[$idx]}
  echo ${exp[$idx]}

done
