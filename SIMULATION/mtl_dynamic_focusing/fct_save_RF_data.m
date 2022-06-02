function [] = fct_save_RF_data(data, pres, folder)
    fname = fullfile(pres, folder, 'RF_data.mat');
    save(fname, 'data')
end