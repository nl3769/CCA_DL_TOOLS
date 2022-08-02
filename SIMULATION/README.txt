Script for realistic simulation of the common carotid artery (CCA) using FIELDII or SIMUS.

Four scripts must be run to get the final image:
1 - mtl_cores/fct_run_mk_phantom.m
2 - mtl_cores/fct_run_wave_propagation.m
3 - mtl_cores/fct_run_cluster_RF.m
4 - mtl_cores/fct_run_image_reconstruction.m

Those scripts are detailed below.

########################################
#### mtl_cores/fct_run_mk_phantom.m ####
########################################

This script generate the phantom(s). It is associated to .json file which contains all the parameters recquired for the simulation.

How to run?

matlab -r fct_run_mk_phantom('$pfolder','$dname','$pres','$pparam','$info')
 pfolder   -> path where static images are stored.
 dname     -> name the image we want to simulate.
 pres      -> path to store result.
 pparam    -> path to the .json file.
 info      -> add information in the name of the experiment.

The result is managed as follow :
 pparam/dname + info/dname + id_sequence + info/raw_data/raw_     -> store simulation results
 pparam/dname + info/dname + id_sequence + info/phantom/          -> store phantom, LI/MA segmentation results, image information 
 pparam/dname + info/dname + id_sequence + info/parameters        -> copy/paste the json file to track parameters
 pparam/dname + info/dname + id_sequence + info/bmode_results/RF  -> store final result in png format and some image for qualitative interpretation

How to run on local computer?
 Run the script run_local/run_mk_phantom.m
How to run on cluster?
 Run the script run_cluster/run_CLUSTER_mk_phantom.py

##############################################
#### mtl_cores/fct_run_wave_propagation.m ####
##############################################

Simulation of each shooting event.

How to run?

matlab -r fct_run_wave_propagation('${path_param}','${path_phantom}',${id_tx})
OR
matlab -r fct_run_wave_propagation('${path_param}','${path_phantom}', ${id_tx_start}, ${id_tx_end})
 path_param     -> path to parameters
 path_phantom   -> path to phantom
 id_tx          -> id of the transmitted transducter
 id_tx_start    -> id of the first transmitted transducter
 id_tx_end      -> id of the last transmitted transducter

Results are stored in raw_data/raw_

How to run on local computer?
 Run the script run_local/run_simulation.m

########################################
#### mtl_cores/fct_run_cluster_RF.m ####
########################################

Gather the results of the simulation in a single matrix saved as .mat

How to run?
matlab -r fct_run_cluster_RF('${path_RF}')
 path_RF -> path to simulation results

How to run on local computer?
 Run the script run_local/run_simulation.m

##################################################
#### mtl_cores/fct_run_image_reconstruction.m ####
##################################################

Generate Bmode image.
How to run?
matlab -r fct_run_image_reconstruction('${pres}')
 pres -> path where results are save


How to run on local computer?
 Run the script run_local/run_beamforming.m
How to run on cluster?
 Run the script run_cluster/run_CLUSTER_beamforming.py

#########################################
#### RUN THE PIPELINE ON THE CLUSTER ####
#########################################

Two step are recquired to run the pipeline:
1) Run run_cluster/run_CLUSTER_mk_phantom.py    -> it creates all the phantom you want (the number of desired patients is controlled in the python code, and the number of sequence in .json file)
2) Run run_cluster/run_CLUSTER_svl_pipeline.py  -> it parallelize the simulation on the cluster, the output is the Bmode image (it uses job array and job dependancies)
