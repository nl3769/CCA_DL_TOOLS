close all; 
clearvars;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% cluster RF data

run(fullfile('..', 'package_utils', 'add_path.m'))

% --- path to data

path_data='/home/laine/Desktop/TEST_SEG/momo/tech_001_id_001_FIELD';
path_data=strcat(path_data, '/raw_data');

fct_run_cluster_RF(path_data)