close all; 
clearvars;


if ~isdeployed
    addpath(fullfile('..', 'mtl_utils'))
    addpath(fullfile('..', 'mtl_class/'))
    addpath(fullfile('..', 'mtl_cores'))
end
 
%%%%%%% RUM SIMULATION FOR ONE PHANTOM AND ONE SET OF PARAMETERS %%%%%%% 

% --- path to data
path = '/home/laine/HDD/PROJECTS_IO/SIMULATION/IMAGENET_STA/n02747177_ashcan';
frame = list_files(path);

for idp=1:1:length(frame)
    path_data = fullfile(path, frame{idp});

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
    
    parfor id_tx=1:1:tx
%     for id_tx=1:1:tx
        % --- run simulation
        fct_run_wave_propagation(fullfile(path_data, 'parameters', parameters_folder{1}), fullfile(path_data, 'phantom', phantom_names{1}), id_tx);
    
    end
end
% fct_cluster_RF(fullfile(path_data, 'raw_data'))

% -------------------------------------------------------------------------
function [files] = list_files(path)

    listing = dir(path);
    incr=1;
    
    for id=1:1:length(listing)
        if listing(id).name ~= '.'
            files{incr} = listing(id).name;
            incr=incr+1;
        end
    end

end
% -------------------------------------------------------------------------