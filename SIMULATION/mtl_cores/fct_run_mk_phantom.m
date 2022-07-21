function fct_run_mk_phantom(varargin)
    
    % --- get parameters
    switch nargin
      
      case 5
        pfolder = varargin{1};
        dname = varargin{2};
        pres = varargin{3};
        pparam = varargin{4};
        info = varargin{5};
      
        otherwise
        error('Problem with parameters (fct_run_wave_propagation)')
    
    end
   

    if ~isdeployed
        run(fullfile('..', 'mtl_utils', 'add_path.m'));
    end
    
    % --- load parameters
    parameters = parametersHandler(pparam);
    
    ndname = remove_extension(dname);
    pname = strcat('dicom_', ndname, '_phantom_', parameters.param.soft, info);
    pres_ = fullfile(pres, strcat(ndname, '_', parameters.param.soft, info));
    pdata = fullfile(pfolder, dname);
    
    % --- set parameters
    parameters.set_pres(pres_);                                      % TODO
    parameters.set_path_data(pdata);                                 % TODO
    parameters.set_phantom_name(pname);                              % TODO
    parameters.create_directory();                                   % TODO
    parameters.save();                                               % TODO

    % --- make phantom
    phantom = createPhantom(pres_, get_extension(dname), [1,0,1,0]);
    phantom.get_image(); 
    phantom.get_scatteres_from_img(); 
%     phantom.phantom_tmp();
    phantom.extrusion(true);    
    phantom.remove_top_region(parameters.param.remove_top_region);
    phantom.init_position();
    substr = fct_get_substr_id_seq(phantom.id_seq);
    pname = strcat('dicom_', ndname, '_phantom_' , 'id_', substr , '_', parameters.param.soft, info);
    pres_ = fullfile(pres, strcat(ndname, '_' , 'id_', substr , '_', parameters.param.soft, info));
    % --- remove first phnatom which is not used
    rmdir(parameters.param.path_res, 's')
    % --- update parameters
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
        
    for id_img = 1:1:parameters.param.nb_images
        
        phantom.update_parameters(pres_);                    
        phantom.animate_scatteres();
        phantom.clear_moved_image();
        phantom.remove_top_region(parameters.param.remove_top_region);
        phantom.create_OF_GT(id_img);
        phantom.save_image();
         phantom.save_scatteres();

        phantom.incr_id_sequence();

        substr = fct_get_substr_id_seq(phantom.id_seq);
        
        if id_img < parameters.param.nb_images

            pname = strcat('dicom_', ndname, '_phantom_' , 'id_', substr , '_', parameters.param.soft, info);
            pres_ = fullfile(pres, strcat(ndname, '_' , 'id_', substr , '_', parameters.param.soft, info));
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
    % Remove extebsion of a filename. 

    ndname = erase(dname, ".tiff");
    ndname = erase(ndname, ".png");
    ndname = erase(ndname, ".jpg");
    ndname = erase(ndname, ".JPEG");
    ndname = erase(ndname, ".DICOM");
end

% -------------------------------------------------------------------------
function [type] = get_extension(dname)
    % Get the extension of a filename.

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
