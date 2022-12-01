% ------------------------------------------------------------------------------------------------------------------------------
function [probe, sub_probe, param, phantom]=fct_get_data(path_data, param_name, probe_name, sub_probe_name, phantom_name)
    % Load data required for image reconstruction.
    
    % --- PARAMETERS
    param=fct_load_param(fullfile(path_data, 'parameters', param_name));
    disp('Parameters are loaded');
    % --- PROBE 
    probe=load(fullfile(path_data, 'raw_data', probe_name));
    probe=probe.probe;
    disp('Probe data is loaded');
    
    % ---  SUBPROBE 
    sub_probe=load(fullfile(path_data, 'raw_data', sub_probe_name));
    sub_probe=sub_probe.sub_probe;
    disp('Probe data is loaded');
    
    % --- PHANTOM
    phantom=fct_load_phantom(fullfile(path_data, 'phantom', phantom_name));
    disp('Phantom is loaded');
    
end


