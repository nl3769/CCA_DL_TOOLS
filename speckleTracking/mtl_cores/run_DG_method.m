function run_DG_method(varargin)
    
    % --- get parameters
    switch nargin
      
      case 2
        pdata = varargin{1};
        pres = varargin{2};
        otherwise
        error('Problem with parameters (fct_run_mk_phantom)')
    end
    
    
    % --- list number of patients
    files = dir(pdata);
    nb_files = size(files, 1);
    start_id = 3;
    
    % --- known subfold
    toBmode = 'bmode_result/results';
    toMotion = 'phantom';
    toImage = 'phantom';
    toParam = 'parameters';
    
    f=figure('visible', 'off');
    
    set(gcf, 'Position', get(0, 'Screensize'));
    set(get(gcf, 'Children'), 'Visible', 'off'); % axis of for each image
    
    % --- loop over patients
    for id=start_id:nb_files
        
        patient_name = fullfile(files(id).folder, files(id).name);
        pres_img = fullfile(pres, files(id).name, 'images_res');
        pres_num = fullfile(pres, files(id).name, 'num_res');
        
        mkdir(pres_img)
        mkdir(pres_num)
        
        % --- get path to data
        simu = dir(patient_name);
        
        pI1 = fullfile(patient_name, simu(3).name, toBmode);
        pI2 = fullfile(patient_name, simu(4).name, toBmode);
        pcf = fullfile(patient_name, simu(3).name, toImage);
        pparam = fullfile(patient_name, simu(3).name, toParam);
        
        pmotion = fullfile(patient_name, simu(4).name, toMotion);
        
        pI1 = fct_get_path_from_substr(pI1, '_bmode.png', dir(pI1));
        pI2 = fct_get_path_from_substr(pI2, '_bmode.png', dir(pI2));
        pmotion = fct_get_path_from_substr(pmotion, '.nii', dir(pmotion));
        pparam = fct_get_path_from_substr(pparam, '.json', dir(pparam));
        pcf = fct_get_path_from_substr(pcf, 'image_information', dir(pcf)); 
        
        % --- get seq
        I1 = imread(pI1);
        I2 = imread(pI2);
        seq = cat(3, I1, I2);
        motion_gt = niftiread(pmotion);
        img_info = load(pcf);
        cf = img_info.image.CF;
        param = fct_load_param(pparam);
        z_start = 2 * param.remove_top_region;
        motion_gt = fct_adapt_flow(I1, motion_gt, cf, z_start);
        
        [Dz, Dx, id_z, id_x] = fct_compute_motion_GT(seq);
        [Dz_gt, Dx_gt] = fct_get_gt(id_x, id_z, motion_gt(:, :, :));
        
        
        fct_mk_figure(Dx, Dz, Dx_gt, Dz_gt, pres_img, f);
        fct_save_res(id_x, id_z, Dx, Dz, Dx_gt, Dz_gt, pres_num);
        fct_save_CF(cf, pres_num);
    end
    close(f);

end