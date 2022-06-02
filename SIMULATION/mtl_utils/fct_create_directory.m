function [path_res]=fct_create_directory(path_data, name)
    % Create directory containing the name of the raw_data if does not
    % exist
    
    % --- remove raw_data.mat, think to respect the form of path_data
    name=erase(name,'_raw_data.mat');
    path_res=fullfile(path_data, 'bmode_result', name);
    % --- create directory if does not exist
    if ~exist(path_res, 'dir')
        mkdir(path_res);
    end

end