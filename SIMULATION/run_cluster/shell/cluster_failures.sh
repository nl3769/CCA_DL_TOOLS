#!/bin/bash


echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo '""""""""""""" LAUNCH SIMULATION FULL PIPELINE """"""""""""""""'
echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo''

echo '""""""""""""""""""""""""" PARAMETERS """"""""""""""""""""""""""'

echo "path param:       $1"      # path_param
echo "path phantom:     $2"      # path_phantom
echo "log name:         $3"      # log_name
echo "id_tx:            $4"      # id_tx

echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

# --- RUN SIMULATION
SIMULATION=$(qsub -N $4 -v path_param=$1,log_name=$3,path_phantom=$2,id_tx=$4 pbs/simulation_failures.pbs)
echo $SIMULATION