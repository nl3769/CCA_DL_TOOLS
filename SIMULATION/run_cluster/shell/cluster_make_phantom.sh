#!/bin/bash


echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
echo '"""""""""""""     CREATE PHANTOM CLUSTER       """""""""""""""'
echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

echo '""""""""""""""""""""""""" PARAMETERS """"""""""""""""""""""""""'

echo "pfolder:   $1"          
echo "dname:     $2"           
echo "pres:      $3"             
echo "pjson:     $4" 
echo "info:      $5"             

echo '""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
# --- generate phantom
PHANTOM_CREATION=$(qsub -N $2 -v pfolder=$1,dname=$2,pres=$3,pparam=$4,info=$5 pbs/make_phantom.pbs)
echo SPHANTOM_CREATION

