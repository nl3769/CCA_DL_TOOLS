#!/bin/bash


echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo '""""""""""""" LAUNCH SIMULATION FULL PIPELINE """"""""""""""""'
echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo''

echo '""""""""""""""""""""""""" PARAMETERS """"""""""""""""""""""""""'

echo "path param: 		$1"      # path_param
echo "path phantom:  		$2"      # path_phantom
echo "flag cluster: 		$3"      # flag_cluster
echo "log name: 		$4"      # log_name
echo "id_tx: 			$5"      # id_tx
echo "path_raw_data: 		$6"      # path raw_data
echo "path res: 		$7"      # pres

echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

# --- RUN SIMULATION
SIMULATION=$(qsub -N $4 -v path_param=$1,path_phantom=$2,flag_cluster=$3,log_name=$4,id_tx=$5 pbs/simulation.pbs)
echo $SIMULATION

# --- CLUSTER RF DATA
cls_RF_name="$4_CLS"
CLUSTER_RF=$(qsub -N $cls_RF_name -v path_RF=$6,log_name=$4 -W depend=afterokarray:$SIMULATION pbs/cluster_RF.pbs)
#CLUSTER_RF=$(qsub -N $cls_RF_name -v path_RF=$6,log_name=$4 ../pbs_scripts/CLUSTER_run_cluster_RF.pbs)
echo $CLUSTER_RF


# --- BEAMFORM DATA
bf_name="$4_BF"
BEAMFORMING=$(qsub -N $bf_name -v log_name=$4,flag_cluster=$3,pres=$7 -W depend=afterany:$CLUSTER_RF pbs/beamforming.pbs)
#BEAMFORMING=$(qsub -N $bf_name -v log_name=$4,flag_cluster=$3,pres=$7 ../pbs_scripts/CLUSTER_run_beamforming.pbs)
echo $BEAMFORMING
