#!/bin/bash

parameters=("set_parameters_training_seg_fold0.py"
            "set_parameters_training_seg_fold1.py"
            "set_parameters_training_seg_fold2.py"
            "set_parameters_training_seg_fold3.py"
            "set_parameters_training_seg_fold4.py"
            "set_parameters_training_seg_fold5.py"
            "set_parameters_training_seg_fold6.py"
            "set_parameters_training_seg_fold7.py"
            "set_parameters_training_seg_fold8.py"
            "set_parameters_training_seg_fold9.py"
            )

exp=("fold0"
     "fold1"
     "fold2"
     "fold3"
     "fold4"
     "fold5"
     "fold6"
     "fold7"
     "fold8"
     "fold8"
     )



for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} training_seg.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
