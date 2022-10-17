#!/bin/bash

parameters=("set_parameters_training_flyingChairs_raft.py"
            "set_parameters_training_flyingChairs_gma.py")

exp=("decod_GRU"
     "decod_transformer")



for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} training_flow.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
