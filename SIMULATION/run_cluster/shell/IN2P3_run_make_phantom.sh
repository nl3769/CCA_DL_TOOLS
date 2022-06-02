#!/bin/bash


echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo '"""""""""""""      CREATE PHANTOM IN2P3        """""""""""""""'
echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

echo '""""""""""""""""""""""""" PARAMETERS """"""""""""""""""""""""""'

echo "pfolder: $1"          
echo "dname: $2"           
echo "pres: $3"             
echo "info: $4"             
echo "soft: $5"             
echo "acq_mode: $6"         
echo "number of images in the sequence: $7"
plog=/sps/creatis/nlaine/PROJECTS_IO/log/MAKE_PHANTOM

echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
# --- generate phantom
PHANTOM_CREATION=$(sbatch --job-name=$2 --output=$plog/$2 --export=pfolder=$1,dname=$2,pres=$3,info=$4,soft=$5,acq_mode=$6,nb_img=$7 ../slurm_scripts/IN2P3_make_phantom.slurm)
echo SPHANTOM_CREATION
