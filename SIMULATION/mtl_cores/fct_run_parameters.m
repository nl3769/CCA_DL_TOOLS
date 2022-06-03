function fct_run_parameters(varargin)
    
    % --- get parameters
    switch nargin
      
      case 7
        pfolder = varargin{1};
        dname = varargin{2};
        pres = varargin{3};
        info = varargin{4};
        soft = varargin{5};
        acq_mode = varargin{6};
        nb_img = varargin{7};
        roi = [1, 0, 1, 0];
      
      case 11 
        pfolder = varargin{1};
        dname = varargin{2};
        pres = varargin{3};
        info = varargin{4};
        soft = varargin{5};
        acq_mode = varargin{6};
        nb_img = varargin{7};
        
        if ischar(varargin{8})
           x_start = str2double(varargin{8});
        else
           x_start = varargin{8};
        end
       
        if ischar(varargin{9})
           x_end = str2double(varargin{9});
        else
           x_end = varargin{9};
        end

        if ischar(varargin{10})
           y_start = str2double(varargin{10});
        else
           y_start = varargin{10};
        end
       
        if ischar(varargin{11})
           y_end = str2double(varargin{11});
        else
           y_end = varargin{11};
        end

        roi = [x_start, x_end, y_start, y_end];
      
        otherwise
        error('Problem with parameters (fct_run_wave_propagation)')
    
    end

    if isstring(nb_img)
        nb_img = str2double(nb_img);
    end
    

    if ~isdeployed
        run(fullfile('..', 'mtl_utils', 'add_path.m'));
    end

    software = soft; % SIMUS, FIELD
    slice_spacing = 0.5e-4;
    nb_slice = 10;
    acquisition_mode = acq_mode; % scanline_based, synthetic_aperture 
    scat_density = 0.02;
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
    parameters.set_dynamic_focusing(1);
    parameters.set_compensation_time(-1);
    parameters.create_directory();
    parameters.save();

    
    % ---------------------------------------------------------------------
    % ------------------------- CREATE PHANTOM ----------------------------
    % ---------------------------------------------------------------------
    
    phantom = createPhantom(pres_, get_extension(dname), roi);
    phantom.get_image(); 
    phantom.get_scatteres_from_img(); 
    phantom.extrusion(true);    
    phantom.remove_top_region(parameters.param.remove_top_region);
    
%     phantom.phantom_tmp();
    
    phantom.init_position();
    substr = fct_get_substr_id_seq(phantom.id_seq);
    pname = strcat('dicom_', ndname, '_phantom_' , 'id_', substr , '_', software, info);
    pres_ = fullfile(pres, strcat(ndname, '_' , 'id_', substr , '_', software, info));
    parameters.set_pres(pres_);
    parameters.set_phantom_name(pname);
    parameters.create_directory()
    parameters.save()
    
    % --- get path to segmentation
    str_=fct_build_path(split(pfolder, '/'), 1);
    str_ = fullfile(str_, "SEG/");
    patient_name = split(dname, '.');
    patient_name = patient_name{1};
    pLI = fullfile(str_, strcat(patient_name, "_IFC3_A1.txt"));
    pMA = fullfile(str_, strcat(patient_name, "_IFC4_A1.txt"));
    phantom.get_seg(pLI, pMA)
    
    for id_img = 1:1:nb_images
        
        phantom.update_parameters(pres_);                    
        phantom.get_gaussian_parameters(20);
        phantom.animate_scatteres();
        phantom.clear_moved_image();
        phantom.remove_top_region(parameters.param.remove_top_region);
        phantom.create_OF_GT(id_img);
        phantom.save_image();
        phantom.save_scatteres();
        phantom.incr_id_sequence();

        substr = fct_get_substr_id_seq(phantom.id_seq);
        
        if id_img < nb_images

            pname = strcat('dicom_', ndname, '_phantom_' , 'id_', substr , '_', software, info);
            pres_ = fullfile(pres, strcat(ndname, '_' , 'id_', substr , '_', software, info));
            parameters.set_pres(pres_);
            parameters.set_phantom_name(pname);
            parameters.create_directory()
            parameters.save()

            phantom.update_parameters(pres_);
        end
        
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
