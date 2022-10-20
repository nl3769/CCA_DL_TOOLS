#!/bin/bash

parameters=(
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            "set_parameters_training_GMA_fine_tuning.py"
            )

exp=(
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     "FT_GMA"
     )



for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} training_flow.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
