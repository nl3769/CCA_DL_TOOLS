#!/bin/bash

parameters=("GAN_parameters_L1_GAN1_PIXEL1.py"
            "GAN_parameters_L2_GAN1_PIXEL1.py"
            "GAN_parameters_L1L2_GAN1_PIXEL1.py"
            "GAN_parameters_L1_GAN1_PIXEL5.py"
            "GAN_parameters_L2_GAN1_PIXEL5.py"
            "GAN_parameters_L1L2_GAN1_PIXEL5.py"
            )

exp=("L1_D1_P1"
     "L2_G1_P1"
     "L1L2_G1_P1"
     "L1_G1_P5"
     "L2_G1_P5"
     "L1L2_G1_P5"
     )


for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} run_cluster.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
