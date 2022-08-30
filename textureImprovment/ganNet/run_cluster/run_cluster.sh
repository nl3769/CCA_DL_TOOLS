#!/bin/bash

parameters=(
            "Unet_parameters_cluster_L1_upconv.py"
            "Unet_parameters_cluster_L2_upconv.py"
            )

exp=(
    "upl1_255"
    "upl2_255"
     )


for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} run_cluster.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
