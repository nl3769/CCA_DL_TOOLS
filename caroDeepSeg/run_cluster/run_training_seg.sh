#!/bin/bash

parameters=("set_parameters_training_seg_f0.py"
            "set_parameters_training_seg_f1.py"
            "set_parameters_training_seg_f2.py"
            "set_parameters_training_seg_f3.py"
            "set_parameters_training_seg_f4.py"
            "set_parameters_training_seg_f5.py"
            "set_parameters_training_seg_f6.py"
            "set_parameters_training_seg_f7.py"
            "set_parameters_training_seg_f8.py"
            "set_parameters_training_seg_f9.py"
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
     "fold9"
     )



for idx in ${!parameters[@]}; do
  qsub -N ${exp[$idx]} -v log_name=${exp[$idx]},param=${parameters[$idx]} training_seg.pbs
  echo ${parameters[$idx]}
  echo ${exp[$idx]}
done
