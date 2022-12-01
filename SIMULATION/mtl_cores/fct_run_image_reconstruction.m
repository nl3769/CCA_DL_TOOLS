function fct_run_image_reconstruction(pres)
    close all;
    % --- add path
    if ~isdeployed
        run(fullfile('..', 'mtl_utils', 'add_path.m'))
    end
    
    praw_data = fullfile(pres, 'raw_data');
    [RF_signals, nb_tx] = fct_run_cluster_RF(praw_data);
    if nb_tx == 192
        % --- get .mat in folders
        raw_data_folder   = fct_list_ext_files(pres, 'mat', 'raw_data');
        parameters_folder = fct_list_ext_files(pres, 'json','parameters');
        phantom_folder    = fct_list_ext_files(pres, 'mat', 'phantom');
        % --- get data in raw_data folder
        probe_name      = fct_detect_sub_str(raw_data_folder, 'probe.mat');
        sub_probe_name  = fct_detect_sub_str(raw_data_folder, 'subProbe.mat');    
        % --- get data in phantom folder
        phantom_name = fct_detect_sub_str(phantom_folder, 'dicom');
        % ---  create object 
        data = imageReconstruction(pres, parameters_folder{1}, probe_name{1}, sub_probe_name{1}, phantom_name{1}, RF_signals);
        % --- beamforming
        if data.param.mode(1) % scanline based
            data.DAS_scanline_based('DAS');
            data.get_bmode_gamma_correction();
            data.scan_conversion();

        elseif data.param.mode(2) && strcmp(data.param.soft, 'FIELD') == 1 && data.param.dynamic_focusing == 1 % dynamic acuiqistion

            data.init_using_DA();
            data.get_bmode_gamma_correction();
            data.scan_conversion();

        elseif data.param.mode(2) % synthetic aperture
            data.BF_CUDA_STA('DAS');
    %         data.BF_DG()
    %         data.BF_CUDA_STA('DMAS');

    %         data.DAS_synthetic_aperture('DAS', 'SUM');        
            data.get_bmode_gamma_correction();

        end

        data.adapt_dimension();
        data.postprocessing();
        [pres_in_vivio, x_disp, z_disp] = data.save_in_vivo();
        data.save_beamformed_data();
        [psave, pres_sim] = data.save_bmode(true);    
%         fct_analysis(psave, pres_in_vivio, pres_sim, x_disp, z_disp);
    else
        disp('Problem in the number of received RF signals');
    end
    
end

% -------------------------------------------------------------------------
function [path] = get_org_img_name(pres)
    pI = fullfile(pres, '..', '..', 'phantom');
    listfiles = dir(pI);
    dim = length(listfiles);

    for i=1:1:dim 
        name = listfiles(i).name;
        if contains(name, 'original')
            if contains(name, 'physical_dimension') == 0
                path = fullfile(pI, name);
            end
        end
    end
end

