# Overview

The following REPO provides codes for the following topics:
* SIMULATION
* caroDeepSeg
* caroDeepMotion
* speckleTracking
* textureImprovment

## SIMULATION
All this code run in *matlab* and *cuda*. For one simulation, the process is as follow:
1. make phantom
2. simulation
3. beamforming

The whole process can be performed locally, but the simulation step is very time consuming. Therefore, the phantom and simulation are performed on the CREATIS cluster and the beamforming can be performed locally, provided you have a GPU on your computer.

## run the code locally
The functions to run are in *SIMULATION/run_local*. It contains several functions:
* run_mk_phantom.m
* run_simulation.m
* run_beamforming.m

### SIMULATION/run_local/run_mk_phantom.m
This function call the another function *mtl_cores/fct_run_mk_phantom* which takes as input the path to the database (path to the images), the path to save the phantom, the path to the simulation parameter (.json file) and some additional information (extra_info) to write in the name of the saved file.

The output is structured as follow:
* path_res/img_name/img_name_id_001_extra_info/bmode_result: path to save the bmode data during beamforming
* path_res/img_name/img_name_id_001_extra_info/parameters: copy of the .json to track the parameters
* path_res/img_name/img_name_id_001_extra_info/phantom: save the phantom in .mat format
* path_res/img_name/img_name_id_001_extra_info/raw_data: path save the raw data during simulation

### SIMULATION/run_local/run_simulation.m
This function has to be run after once the phantom is created. It calls the function *mtl_cores/fct_run_wave_propagation.m*. This one takes as argument the path to the phantom, the path to the parameters and the id of the transmitted element. Then the function writes the radiofrequency signal in path_res/img_name/img_name_id_001_extra_info/raw_data/_raw. 

### SIMULATION/run_local/run_beamforming.m
This function has to be run at the end. It calls the function *mtl_cores/fct_run_image_reconstruction.m*. The beamforming is performed on GPU.

#### How tu run on GPU?
To run the beamforming on GPU, first check your GPU's architecture. For linux user, run in a terminal:
```sh
nvidia-smi -q | grep Architecture
```
Then you have to compile the '*.cu*' locating in SIMULATION/cuda/*.cu. For this, for Pascal architecture, run:
```sh
bash compile_PASCAL.sh
```
You can easily create a new compile.sh for a different architecture by modidying the sm_xy number by looking in the following [link](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation).

## run the code on cluster

The cluster uses *PBS Scheduler*. Once your logged to the cluster, each function is handle by python scripts in *SIMULATION/run_cluster/*. 

### SIMULATION/run_cluster/run_CLUSTER_mk_phantom.py

Change the parameters in *run_CLUSTER_mk_phantom.py*, and it will make the phantom for each images in the datase. It can *TIFF*, *JPEG*, *DICOM*.

### SIMULATION/run_cluster/run_mk_phantom.m



### SIMULATION/run_cluster/run_mk_phantom.m


# caroDeepSeg

# caroDeepMotion

# speckleTracking

# textureImprovment
