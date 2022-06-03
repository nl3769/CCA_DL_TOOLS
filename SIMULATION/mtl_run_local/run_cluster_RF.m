close all; 
clearvars;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% cluster RF data

run(fullfile('..', 'mtl_utils', 'add_path.m'))
addpath(fullfile('..', 'mtl_cores'))
% --- path to data

path_data='/home/laine/cluster/PROJECTS_IO/SIMULATION/CUBS/tech_008/tech_008_id_010_FIELD_3D';
path_data=strcat(path_data, '/raw_data');

fct_run_cluster_RF(path_data)