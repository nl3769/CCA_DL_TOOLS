function[]=fct_generate_phantom(path_param_)

    % --- add path
    run('add_path.m')

    files=list_mat_files(path_param_, 'parameters/');

    % --- we generate only one phantom using a random probe (only the probe is common in the parameters)
    path_param=fullfile(path_param_, 'parameters', files{1});

    % --- we create the phantom
    createPhantom(path_param);

end