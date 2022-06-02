#!/bin/bash


echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo '"""""""""""""     CREATE PHANTOM CLUSTER       """""""""""""""'
echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

echo '""""""""""""""""""""""""" PARAMETERS """"""""""""""""""""""""""'

echo "pfolder: 					$1"          
echo "dname: 					$2"           
echo "pres: 					$3"             
echo "info: 					$4"             
echo "soft: 					$5"             
echo "acq_mode: 				$6"         
echo "number of images in the sequence: 	$7"

echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
# --- generate phantom
PHANTOM_CREATION=$(qsub -N $2 -v pfolder=$1,dname=$2,pres=$3,info=$4,soft=$5,acq_mode=$6,nb_img=$7 pbs/make_phantom.pbs)
echo SPHANTOM_CREATION

