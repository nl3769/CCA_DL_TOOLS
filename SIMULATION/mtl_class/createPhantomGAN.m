classdef createPhantomGAN < handle   
    
    properties (Access = private)
        
        type;
        flag_manu;          % (bool) to know if the the phantom is manually created
        param;
        F;                  % object to interpolate scatterers
        x_scat_pos;
        z_scat_pos;    

    end
    
    properties
        data_img;           % (struct) contain the scatterers maps and other information
        data_scatt;         % (struct) contain the scatterers maps and other information
        data_scatt_moved;   % (struct) contain the scatterers maps and other information
        id_seq;             % id of the image in the simulated sequence
        
        scatt_pos_ref;
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj = createPhantomGAN(path_param, type, nb_images)
            % constructor %
                        
%             obj.param=fct_load_param(path_param, false);
            
            % --- load parameters and store them in a structure
            list_files_param=fct_list_mat_files(path_param, 'parameters');

            for id_param=1:1:size(list_files_param, 2)
                obj.param{id_param}=fct_load_param(fullfile(path_param, 'parameters', list_files_param{id_param}), false);
            end
            disp('Parameters are loaded');
                        
            % --- set the type of the image
            obj.type = type;
            obj.flag_manu = false;
            
            % --- load image
            switch obj.type
                case 'DICOM'
                    obj.data_img=load_dicom(obj.param{id_param}.path_data, nb_images);
                                              
            otherwise
            	disp('Only DICOM')
            end
                        
            obj.id_seq = 1;

        end
        % ----------------------------------------------------------------------------------------------------------------------
        
        function get_scatteres_position(obj)

            % --- compute the physical dimension in meter
            height_meter=obj.data_img.height*obj.data_img.CF;
            width_meter=obj.data_img.width*obj.data_img.CF;

            % --- get parameters
            addpath(fullfile('..', 'ext_libraries', 'MUST_2021')')
            probe=getparam(obj.param{1}.probe_name);
            lambda=obj.param{1}.c/probe.fc;
            nb_scatteres_speckle=round(obj.param{1}.scat_density*height_meter*width_meter/(lambda^2));

            % --- create cartezian grid for interpolation
            x_axis=linspace(0, obj.data_img.width, obj.data_img.width);
            z_axis=linspace(0, obj.data_img.height, obj.data_img.height);
            
            % --- generate n random positions
            obj.x_scat_pos=random('unif', 0, obj.data_img.width, fix(nb_scatteres_speckle), 1);
            obj.z_scat_pos=random('unif', 0, obj.data_img.height, fix(nb_scatteres_speckle), 1);

        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function get_scatteres_from_img(obj, id_img)
            % Generates scatterers.
            
            % --- compute the physical dimension in meter
            height_meter=obj.data_img.height*obj.data_img.CF;
            width_meter=obj.data_img.width*obj.data_img.CF;

            % --- compute image axes and the dimension of the phantom (meter)
            x_min=-width_meter/2;
            x_max=width_meter/2;
            z_min=0;
            z_max=height_meter;
            
            x_axis=linspace(0, obj.data_img.width, obj.data_img.width);
            z_axis=linspace(0, obj.data_img.height, obj.data_img.height);
            [x_m,z_m]=meshgrid(x_axis,z_axis);

            % --- get image
            I = obj.data_img.image(:,:,id_img);
            max_val = double(max(I(:)));
            % --- select the desired distribution
            if isequal(obj.param{1}.distribution, [1 0 0]) % simple interpolation
                corrected_B_mode=(double(I)/max_val).^(1/obj.param{1}.gamma);
            elseif isequal(obj.param{1}.distribution, [0 1 0]) % interpolation times normal distribution
                corrected_B_mode=(double(I)/max_val).^(1/obj.param{1}.gamma).*randn(size(obj.data_img.image(:,:,id_img)));
            elseif isequal(obj.param{1}.distribution, [0 0 1]) % interpolation times rayleight distribution
                corrected_B_mode=(double(I)/max_val).^(1/obj.param{1}.gamma).*raylrnd(1,obj.data_img.height,obj.data_img.width)/sqrt(pi/2);
            end
            % --- inverse operation of the log compression
            interpolant=scatteredInterpolant(x_m(:),z_m(:),corrected_B_mode(:), 'linear');

            % --- get the reflexion coefficient
            RC_scat=interpolant(obj.x_scat_pos, obj.z_scat_pos);
            
            % --- convert x_scat and z_scatt in m
            x_scat=obj.x_scat_pos*obj.data_img.CF-obj.data_img.width*obj.data_img.CF/2;         % we center x_scatt around 0
            z_scat=obj.z_scat_pos*obj.data_img.CF;
            
            % --- store the results
            obj.data_scatt{1}{1}.z_max=z_max+obj.param{1}.shift;
            obj.data_scatt{1}{1}.z_min=z_min;
            obj.data_scatt{1}{1}.x_max=x_max;
            obj.data_scatt{1}{1}.x_min=x_min;
            obj.data_scatt{1}{1}.y_max=0;
            obj.data_scatt{1}{1}.y_min=0;
            obj.data_scatt{1}{1}.x_scatt=x_scat;
            obj.data_scatt{1}{1}.z_scatt=z_scat+obj.param{1}.shift;
            obj.data_scatt{1}{1}.y_scatt=zeros(size(x_scat, 1), 1);
            obj.data_scatt{1}{1}.RC_scatt=RC_scat;
            obj.data_scatt{1}{1}.depth_of_focus=3/4*(z_max)+obj.param{1}.shift;

        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function save_scatteres(obj)
            
            % Save scatterers map in png format and in .mat file.
            fct_save_scatterers_2D(obj.data_scatt{1}{1}, obj.param{1}, '_phantom');                  

        end

        % ----------------------------------------------------------------------------------------------------------------------
        function phantom_tmp(obj)

            k=5;
%             obj.data_scatt{1}{1}.z_max=obj.data_scatt{1}{1}.z_max*0.3
%             obj.data_scatt{1}{1}.z_min = 0;
%             obj.data_scatt{1}{1}.x_min=obj.data_scatt{1}{1}.x_min/10;
%             obj.data_scatt{1}{1}.x_max=obj.data_scatt{1}{1}.x_max/10;
            
            z_pos=linspace(obj.data_scatt{1}{1}.z_max*0.1, obj.data_scatt{1}{1}.z_max*0.9, k);
            x_pos=linspace(obj.data_scatt{1}{1}.x_min*0.6, obj.data_scatt{1}{1}.x_max*0.6, k);
            obj.data_scatt{1}{1}.y_scatt=[];
            obj.data_scatt{1}{1}.x_scatt=[];
            obj.data_scatt{1}{1}.z_scatt=[];
            obj.data_scatt{1}{1}.RC_scatt=[];

            for i=1:1:k
                for j=1:k
                    obj.data_scatt{1}{1}.y_scatt=[obj.data_scatt{1}{1}.y_scatt; 0];
                    obj.data_scatt{1}{1}.z_scatt=[obj.data_scatt{1}{1}.z_scatt; z_pos(j)];
                    obj.data_scatt{1}{1}.x_scatt=[obj.data_scatt{1}{1}.x_scatt; 0];
                    obj.data_scatt{1}{1}.RC_scatt=[obj.data_scatt{1}{1}.RC_scatt; 1];
                end
            end
            
%             for i=1:1:k
%                 for j=1:k
%                 obj.data_scatt{1}{1}.y_scatt=[obj.data_scatt{1}{1}.y_scatt; 0];
%                 obj.data_scatt{1}{1}.z_scatt=[obj.data_scatt{1}{1}.z_scatt; z_pos(j)];
%                 obj.data_scatt{1}{1}.x_scatt=[obj.data_scatt{1}{1}.x_scatt; -x_pos(j)];
%                 obj.data_scatt{1}{1}.RC_scatt=[obj.data_scatt{1}{1}.RC_scatt; 1];
%                 end
%             end
            obj.data_scatt{1}{1}.z_scatt=obj.data_scatt{1}{1}.z_scatt+obj.param{1}.shift;
            obj.data_scatt{1}{1}.depth_of_focus=max(obj.data_scatt{1}{1}.z_scatt);

            obj.data_scatt_moved = obj.data_scatt{1}{1};

        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function remove_top_region(obj, dist)
            
            idx = find(obj.data_scatt{1}{1}.z_scatt > dist);
            obj.data_scatt{1}{1}.z_scatt=obj.data_scatt{1}{1}.z_scatt(idx);
            obj.data_scatt{1}{1}.x_scatt=obj.data_scatt{1}{1}.x_scatt(idx);
            obj.data_scatt{1}{1}.y_scatt=obj.data_scatt{1}{1}.y_scatt(idx);
            obj.data_scatt{1}{1}.RC_scatt=obj.data_scatt{1}{1}.RC_scatt(idx);
            
        end

        % ----------------------------------------------------------------------------------------------------------------------
        function extrusion(obj)
            slice_spacing=obj.param{1}.slice_spacing;
            iter = obj.param{1}.nb_slices;
            
            x_img = obj.data_img.width*obj.data_img.CF;
            z_img = obj.data_img.height*obj.data_img.CF;

            pos_z_min=0;
            pos_z_max=z_img;
            pos_x_min=-x_img/2;
            pos_x_max=x_img/2;
            
            nb_scat=length(obj.data_scatt{1}{1}.x_scatt);
                        
            % --- copy of scatterers in 2D plan
            z_scatt = obj.data_scatt{1}{1}.z_scatt;
            x_scatt = obj.data_scatt{1}{1}.x_scatt;
            RC_scatt = obj.data_scatt{1}{1}.RC_scatt;

            for i=1:iter
                % --- compute y-position 
                pos = i*slice_spacing;

                % --- do it twice time for +/- pos
                y_scatt_1 = ones(nb_scat, 1)*pos;                
                y_scatt_2 = -ones(nb_scat, 1)*pos;
                                                
                % --- get new value
                obj.data_scatt{1}{1}.x_scatt = [obj.data_scatt{1}{1}.x_scatt; x_scatt; x_scatt];
                obj.data_scatt{1}{1}.y_scatt = [obj.data_scatt{1}{1}.y_scatt; y_scatt_1; y_scatt_2];
                obj.data_scatt{1}{1}.z_scatt = [obj.data_scatt{1}{1}.z_scatt; z_scatt; z_scatt];
                obj.data_scatt{1}{1}.RC_scatt = [obj.data_scatt{1}{1}.RC_scatt; RC_scatt; RC_scatt];
                
            end
            
            % --- interpolation in 3D space
            y_img = 2*(iter-1)*slice_spacing; %max(obj.data_scatt{1}{1}.y_scatt) - min(obj.data_scatt{1}{1}.y_scatt);
            lambda=obj.param{1}.c/obj.param{1}.fc;
            nb_scat = round(obj.param{1}.scat_density*x_img*y_img*z_img/(lambda^3));
            
            interpolant = scatteredInterpolant(obj.data_scatt{1}{1}.x_scatt, ...
                                               obj.data_scatt{1}{1}.z_scatt, ...
                                               obj.data_scatt{1}{1}.y_scatt, ...
                                               obj.data_scatt{1}{1}.RC_scatt);
                                 
            y_scat = random('unif', -y_img/2, y_img/2, nb_scat, 1);
            x_scat= random('unif', pos_x_min, pos_x_max, nb_scat, 1); 
            z_scat= random('unif', pos_z_min, pos_z_max, nb_scat, 1);
            RC_scatt = interpolant(x_scat, z_scat, y_scat);
            
            % --- update
            obj.data_scatt{1}{1}.x_scatt = x_scat;
            obj.data_scatt{1}{1}.y_scatt = y_scat;
            obj.data_scatt{1}{1}.z_scatt = z_scat;
            obj.data_scatt{1}{1}.RC_scatt = RC_scatt;
            obj.data_scatt{1}{1}.y_min = -y_img/2;
            obj.data_scatt{1}{1}.y_max = y_img/2;
            obj.data_scatt_moved = obj.data_scatt{1}{1};
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function save_image(obj, id_image)
            % Save original image information map and the original image .png.
            
            image = obj.data_img; 
            image.image=image.image(:,:,id_image);
            parameters = obj.param{1};
            % --- save variables in .mat format
            path_res=fullfile(parameters.path_res, 'phantom/image_information.mat');
            save(path_res, 'image');

            % --- save image in .png format
            [col, row]=size(image.image);
            x_disp=row*image.CF;
            x_disp=-x_disp/2:image.CF:x_disp/2;
            z_disp=0:image.CF:col*image.CF;
            
            f=figure('visible', 'off');
            imagesc(x_disp*1e3, z_disp*1e3, image.image); 
            colormap gray;  
            title('original image');
            xlabel('width of the image in mm')
            ylabel('height of the image in mm')
            saveas(f, strcat(parameters.path_res, ['/phantom/original_physical_dimension_' num2str(id_image) '.png']));
    
            imwrite(image.image, strcat(parameters.path_res, ['/phantom/original_' num2str(id_image) '.png']));

        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function update_parameters(obj, pparam)
            % --- load parameters and store them in a structure
            list_files_param=fct_list_mat_files(pparam, 'parameters');

            for id_param=1:1:size(list_files_param, 2)
                obj.param{id_param}=fct_load_param(fullfile(pparam, 'parameters', list_files_param{id_param}), false);
            end
        end
    end
end

% ------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------

function [struct_image]=load_dicom(path_img, nb_images) 
     
    % --- probe info
    info=dicominfo(path_img);
    rect = [info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0+10,...
            info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0+50,... # 50
            info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1-100,...
            info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1-120];

    % load DICOM image
    % --- we load the information

    image_=dicomread(path_img, 'frame', 1:nb_images); 
    for i=1:1:nb_images
        image(:,:,i)=rgb2gray(imcrop(image_(:,:,:,i),rect));
    end

    figure(1)
    for i=1:1:nb_images
        imagesc(image(:,:,i))
        colormap gray
        pause(0.1)
    end

    % --- calibration factor
    CF=info.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaX*1e-2;

    % --- we fill struct_image
    struct_image.CF=CF;
    struct_image.width=size(image, 2);
    struct_image.height=size(image, 1);
    struct_image.image=image;
                                                                                              
end
% ------------------------------------------------------------------------------------------------------------------------------
