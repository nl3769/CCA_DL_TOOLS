#!/bin/bash

parameters=(
	    "GAN_UPCONV_TEST.py"
)

experience=(
             "GAN_UPCONV_TEST"
	    )

for idx in ${!parameters[@]}; do

  sbatch --export=ALL,param=${parameters[$idx]} -o ${experience[$idx]}.log -J ${experience[$idx]} run_JeanZay.slurm
  echo ${parameters[$idx]}
  echo ${exp[$idx]}

done
