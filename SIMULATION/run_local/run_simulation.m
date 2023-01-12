restoredefaultpath;
close all; 
clearvars;

addpath(fullfile('..', 'mtl_utils'))
addpath(fullfile('..', 'mtl_class/'))
addpath(fullfile('..', 'mtl_cores'))

% --- path to data
path_data='/home/laine/Desktop/SIMULATION_TEST/tech_001/tech_001_id_001_FIELD';
% --- get phantom name
phantom_folder = fct_list_ext_files(path_data, 'mat', 'phantom');
phantom_names = fct_detect_sub_str(phantom_folder, 'dicom');
% --- get parameters name
parameters_folder=fct_list_ext_files(path_data, 'json', 'parameters');
% --- loop over transducer
PARAM = fct_load_param(fullfile(path_data, 'parameters', parameters_folder{1}));
nb_tx = PARAM.Nelements;
nb_active = PARAM.Nactive;

% --- run simulation over tx elements
parfor (id_tx=1:nb_tx, 6)
% for id_tx=1:1:192
    % --- run simulation
    fct_run_wave_propagation(fullfile(path_data, 'parameters', parameters_folder{1}), fullfile(path_data, 'phantom', phantom_names{1}), id_tx);
end