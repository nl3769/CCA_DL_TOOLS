#!/bin/bash

parameters=("GAN_parameters_cluster_dropout_00.py"
            "GAN_parameters_cluster_dropout_10.py"
            "GAN_parameters_cluster_dropout_20.py"
            "GAN_parameters_cluster_dropout_40.py"
            "GAN_parameters_cluster_kernel_3.py"
            "GAN_parameters_cluster_kernel_5.py"
            "GAN_parameters_cluster_kernel_7.py"
            "GAN_parameters_normalization_off.py")

exp=("GAN_DO_00"
     "GAN_DO_10"
     "GAN_DO_20"
     "GAN_DO_40"
     "GAN_KS_3"
     "GAN_KS_5"
     "GAN_KS_7.py"
     "GAN_NORM_OFF")


for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} run_cluster.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
