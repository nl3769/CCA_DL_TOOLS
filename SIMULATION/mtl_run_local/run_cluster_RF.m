close all; 
clearvars;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% cluster RF data

run(fullfile('..', 'mtl_utils', 'add_path.m'))
addpath(fullfile('..', 'mtl_cores'))
% --- path to data

path_data='/home/laine/Desktop/QUASI_RANDOM/tech_001_id_001_FIELD';
path_data=strcat(path_data, '/raw_data');

fct_run_cluster_RF(path_data)