close all; 
clearvars;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% cluster RF data

run(fullfile('..', 'mtl_utils', 'add_path.m'))
addpath(fullfile('..', 'mtl_cores'))
% --- path to data

path_data='/home/laine/Documents/PROJECTS_IO/SIMULATION/DEBUG/STA/HIGHT_DENSITY/ILSVRC2012_test_00000026_da/ILSVRC2012_test_00000026_id_001_FIELD_3D';
path_data=strcat(path_data, '/raw_data');

fct_run_cluster_RF(path_data);