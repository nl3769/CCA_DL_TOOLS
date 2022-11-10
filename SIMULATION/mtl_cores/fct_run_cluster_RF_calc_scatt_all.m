function fct_run_cluster_RF_calc_scatt_all(pres)

    % --- add path
    run(fullfile('..', 'mtl_utils', 'add_path.m'))

    % --- get phantom name
    RF_name = fct_list_ext_files(pres, 'mat', 'raw_');
    RF_name = natsortfiles(RF_name);

    % --- load data
	raw_data_ = load_RF_tx(fullfile(pres, 'raw_', RF_name{1}));
    for i=1:1:size(raw_data_, 3)
        raw_data{i} = raw_data_(:,i,:);
    end
    
    % --- save data
    path_raw_data=fullfile(pres, 'RF_raw_data.mat');

    save(path_raw_data, 'raw_data', '-v7.3'); % flag '-v7.3' to store data larger than 2Go

end


% ----------------------------------------------------------------------------------------------------------------------
function RF_tx = load_RF_tx(path)
    RF_tx = load(path);
    RF_tx = RF_tx.raw_data;
end
