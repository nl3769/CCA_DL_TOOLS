close all; 
clearvars;


if ~isdeployed
    addpath(fullfile('..', 'function/'))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%% RUM SIMULATION FOR ONE PHANTOM AND ONE SET OF PARAMETERS %%%%%%% 

% --- path to data
path_data='/home/laine/Desktop/TEST_SEG//momo/tech_001_id_001_FIELD';

% --- get phantom name
phantom_folder=fct_list_mat_files(path_data, 'phantom');
phantom_names=fct_detect_sub_str(phantom_folder, 'dicom');

% --- get parameters name
parameters_folder=fct_list_mat_files(path_data, 'parameters');

% --- loop over transducer
PARAM = load(fullfile(path_data, 'parameters', parameters_folder{1}));
PARAM = PARAM.p;
nb_tx = PARAM.Nelements;
nb_active = PARAM.Nactive;

tx = 0;
if PARAM.mode(1)
    tx = nb_tx - nb_active + 1;
elseif PARAM.mode(2)
    tx = nb_tx;
end

for id_tx=1:1:128
% for id_tx=10:1:10
    
    % --- run simulation
    fct_run_wave_propagation(fullfile(path_data, 'parameters', parameters_folder{1}), fullfile(path_data, 'phantom', phantom_names{1}), false, id_tx);

end

fct_cluster_RF(fullfile(path_data, 'raw_data'))
