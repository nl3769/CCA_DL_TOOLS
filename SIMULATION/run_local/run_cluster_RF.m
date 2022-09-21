close all; 
clearvars;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% cluster RF data

run(fullfile('..', 'mtl_utils', 'add_path.m'))
addpath(fullfile('..', 'mtl_cores'))
% --- path to data

path_data='/home/laine/Desktop/displacment_test/CAMO01_image1/CAMO01_image1_id_003_FIELD';
path_data=strcat(path_data, '/raw_data');

fct_run_cluster_RF(path_data)