function fct_save_scatterers_3D(data_scatt, param, substr)
        
    % --- save figure -> remove if it is used on VIP platform
%     addpath('../display/')
%     make_figure_3D_scatt(data_scatt, param, substr)
    
    % --- save the numeric phantom in .mat file
    if strcmp(substr, '')
        substr_ = 'original_scatterers_distribution';
    else
        substr_ = param.phantom_name;
    end
    path_ph = strcat(substr_, '.mat');
    phantom_name=fullfile(param.path_res, 'phantom', path_ph);
    scatt=data_scatt;
    save(phantom_name, 'scatt');

end
