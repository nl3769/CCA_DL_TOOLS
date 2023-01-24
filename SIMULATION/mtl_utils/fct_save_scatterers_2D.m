function fct_save_scatterers_2D(data_scatt, param, substr)
    
    % --- save figure -> remove if it is used on VIP platform
    
    if ~isdeployed
        make_figure_2D_scatt(data_scatt, param, substr)
    end
    % --- save the numeric phantom in .mat file
    
    if strcmp(substr, '')
        substr_ = 'original_scatterers_distribution';
    else
        substr_ = param.phantom_name;
    end

    path_ph = strcat(substr_, substr, '.mat');    
    phantom_name=fullfile(param.path_res, 'phantom', path_ph);
    scatt=data_scatt;
    save(phantom_name, 'scatt');

end
