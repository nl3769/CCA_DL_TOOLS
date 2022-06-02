function fct_run_parameters_GAN(pfolder, dname, pres, info, soft, acq_mode, nb_img)
    
    if ~isdeployed
        addpath(fullfile('..', 'class/'));   
    end

    software = soft; % SIMUS, FIELD
    slice_spacing = 2e-4;
    nb_slice = 3;
    acquisition_mode = acq_mode; % scanline_based, synthetic_aperture 
    scat_density = 10;
    Nelement = 192;
    Nactive = 65;
    shift = 0;
    nb_images = str2double(nb_img); % number of images in one sequence
    ndname = remove_extension(dname);
    pname = strcat('dicom_', ndname, '_phantom_', software, info);
    pres_ = fullfile(pres, strcat(ndname, '_', software, info));
    pdata = fullfile(pfolder, dname);
    
    % ---------------------------------------------------------------------
    % -------------------------- SET PARAMETERS ---------------------------
    % ---------------------------------------------------------------------
    
    parameters = writeParameters();
    parameters.set_pres(pres_);
    parameters.set_path_data(pdata);
    parameters.set_phantom_name(pname);
    parameters.set_software(software);
    parameters.set_Nactive(Nactive);
    parameters.set_Nelements(Nelement);
    parameters.set_acquisition_mode(acquisition_mode);
    parameters.set_scatteres_density(scat_density);
    parameters.set_nb_slice(nb_slice);
    parameters.set_slice_spacing(slice_spacing);
    parameters.set_shift(shift);
    parameters.set_dynamic_focusing(0);
    parameters.set_compensation_time(-1);
    pname = strcat('dicom_', ndname, '_phantom_' , 'id_', num2str(1) , '_', software, info);
    pres_ = fullfile(pres, strcat(ndname, '_' , 'id_', num2str(1), '_', software, info));
    parameters.set_phantom_name(pname);
    parameters.set_pres(pres_);
    parameters.create_directory();
    parameters.save();

    % ---------------------------------------------------------------------
    % ------------------------- CREATE PHANTOM ----------------------------
    % ---------------------------------------------------------------------
    

    phantom = createPhantomGAN(pres_, get_extension(dname), nb_images);
    phantom.get_scatteres_position();
    phantom.get_scatteres_from_img(1);
    phantom.remove_top_region(parameters.param.remove_top_region);
    phantom.save_image(1);
    phantom.save_scatteres();
    
    for id_img = 2:1:nb_images
        
        % --- set parameters
        pname = strcat('dicom_', ndname, '_phantom_' , 'id_', num2str(id_img) , '_', software, info);
        pres_ = fullfile(pres, strcat(ndname, '_' , 'id_', num2str(id_img), '_', software, info));
        parameters.set_phantom_name(pname);
        parameters.set_pres(pres_);
        parameters.create_directory();
        parameters.save();
        
        % --- set phantom
        phantom.update_parameters(pres_);
        phantom.get_scatteres_from_img(id_img);
        phantom.remove_top_region(parameters.param.remove_top_region);
        phantom.save_image(id_img);
        phantom.save_scatteres();      
        
    end

end

% -------------------------------------------------------------------------
function [ndname] = remove_extension(dname)
    ndname = erase(dname, ".tiff");
    ndname = erase(ndname, ".png");
    ndname = erase(ndname, ".jpg");
    ndname = erase(ndname, ".JPEG");
    ndname = erase(ndname, ".DICOM");
end

% -------------------------------------------------------------------------
function [type] = get_extension(dname)

    if contains(dname, '.tiff')
        type = 'TIFF';
    elseif contains(dname, '.png')
        type = 'PNG';
    elseif contains(dname, '.JPEG')
        type = 'JPEG';
    elseif contains(dname, '.jpg')
        type = 'JPG';
    elseif contains(dname, '.mat')
        type = 'MAT';
    else
        type = 'DICOM';
    end

end
