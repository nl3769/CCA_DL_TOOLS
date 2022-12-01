restoredefaultpath;
close all; 
clearvars;


if ~isdeployed
    addpath(fullfile('..', 'mtl_utils'))
    addpath(fullfile('..', 'mtl_class/'))
    addpath(fullfile('..', 'mtl_cores'))
end
 
%%%%%%% RUN SIMULATION %%%%%%% 

% --- path to data
path_data='/home/laine/Desktop/CAMO03_image2/CAMO03_image2_id_001_FIELD';
% --- get phantom name
phantom_folder = fct_list_ext_files(path_data, 'mat', 'phantom');
phantom_names = fct_detect_sub_str(phantom_folder, 'dicom');
% --- get parameters name
parameters_folder=fct_list_ext_files(path_data, 'json', 'parameters');
% --- loop over transducer
PARAM = fct_load_param(fullfile(path_data, 'parameters', parameters_folder{1}));
nb_tx = PARAM.Nelements;
nb_active = PARAM.Nactive;

tx = 0;
if PARAM.mode(1)
    tx = nb_tx - nb_active + 1;
elseif PARAM.mode(2)
    tx = nb_tx;
end

stop = false;
parfor (id_tx=1:192, 6)
% for id_tx=1:1:192
    % --- run simulation
    stop = fct_run_wave_propagation(fullfile(path_data, 'parameters', parameters_folder{1}), fullfile(path_data, 'phantom', phantom_names{1}), id_tx);
%     if stop
%         break;
%     end
end

if ~stop
    fct_run_cluster_RF(fullfile(path_data, 'raw_data'))
end