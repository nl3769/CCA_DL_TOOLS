#!/bin/bash

parameters=("set_parameters_sharedFeatures.py"
            "set_parameters_splitFeatures.py")

exp=("FlowDeep_SharedFeatures"
     "FlowDeep_SplitFeatures")



for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} training.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done