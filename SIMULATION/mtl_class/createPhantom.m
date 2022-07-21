classdef createPhantom < handle   
    
    properties (Access = private)
        
        type;               % (char) type image/sequence: DICOM, TIFF...
        flag_manu;          % (bool) to know if the the phantom is manually created
        param;              % (struct) contains parameters for simulation
        F;                  % (obj) object to interpolate scatterers
                
        x_offset_rot;       % (double) random offset for rotation
        y_offset_rot;       % (double) random offset for rotation
        z_offset_rot;       % (double) random offset for rotation
        
        x_offset_scaling;   % (double) random offset for scaling
        y_offset_scaling;   % (double) random offset for scaling
        z_offset_scaling;   % (double) random offset for scaling
        
        x_offset_shearing;  % (double) random offset for shearing
        y_offset_shearing;  % (double) random offset for shearing
        z_offset_shearing;  % (double) random offset for shearing

        x_offset_stretch;   % (double) random offset for stretching
        y_offset_stretch;   % (double) random offset for stretching
        z_offset_stretch;   % (double) random offset for stretching
        
        f_simu;             % (double) central frequency of the probe
        theta_max_rot;      % (double) maximal rotation 
        theta_max_shear;    % (double) maximal shearing
        scaling_coef;       % (double) maximal scaling
        stretch_coef;       % (double) maximal stretching
        time_sample_simu;   % (double) total number of time samples for simulation
        fps;

        gaussian_x;         % (double) gaussian parameters -> unused
        gaussian_y;         % (double) gaussian parameters -> unused
        gaussian_z;         % (double) gaussian parameters -> unused
        
    end
    
    properties
        
        data_img;           % (struct) contain scatterers maps and other information
        data_scatt;         % (struct) contain scatterers maps and other information
        data_scatt_moved;   % (struct) contain scatterers maps and other information
        id_seq;             % (int) id of the current processed image
        roi;                % (array) coordinate to crop image/sequence
        scatt_pos_ref;      % (struct) position of the initial scatteres

        seg_LI_ref;         % (struct) original LI segmentation
        seg_MA_ref;         % (struct) original MA segmentation
        seg_LI;             % (struct) moved LI segmentation
        seg_MA;             % (struct) moved MA segmentation
        
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj = createPhantom(varargin)
            % Constructor %
            
            switch nargin
                case 3
                    path_param=varargin{1};
                    type=varargin{2};
                    obj.roi = varargin{3};
                otherwise
                    disp('Problem with parameters (createPhantom constructor)');
            end
            
            % --- load parameters and store them in a structure
            list_files_param=fct_list_mat_files(path_param, 'parameters');
            obj.param = fct_load_param(fullfile(path_param, 'parameters', list_files_param{1}));
            disp('Parameters are loaded');
                        
            % --- set the type of the image
            obj.type = type;
            obj.flag_manu = false;

        end
        % ----------------------------------------------------------------------------------------------------------------------

        function get_image(obj)
            % --- load image

            switch obj.type
                case 'TIFF'
                    obj.data_img=load_tiff(obj.param.path_data, obj.roi);
                case 'JPEG'
                    obj.data_img=load_JPEG(obj.param.path_data, obj.roi);
                case 'DICOM'
                    obj.data_img=load_dicom(obj.param.path_data, obj.param.sequence);
                                              
            otherwise
            	disp('Image not found')
            end
            
            % --- init sequence
            obj.id_seq = 1;
            obj.init_scatt_reference()

        end

        % ----------------------------------------------------------------------------------------------------------------------
        function get_scatteres_from_img(obj)
            % Generate scatterers.
            
            % --- compute the physical dimension in meter
            height_meter = obj.data_img.height*obj.data_img.CF;
            width_meter = obj.data_img.width*obj.data_img.CF;
            % --- compute image axes and the dimension of the phantom (meter)
            x_min=-width_meter/2;
            x_max=width_meter/2;
            z_min=0;
            z_max=height_meter;
            probe=getparam(obj.param.probe_name);
            lambda=obj.param.c/probe.fc;
            nb_scatteres_speckle=round(obj.param.scat_density*height_meter*width_meter/(lambda^2));
            % --- create cartezian grid for interpolation
            x_axis=linspace(0, obj.data_img.width, obj.data_img.width);
            z_axis=linspace(0, obj.data_img.height, obj.data_img.height);
            [x_m,z_m]=meshgrid(x_axis,z_axis);
            % --- apply bilateral filtering
            if obj.param.preprocessing
                save_surf_image(obj.data_img.image, '_before_post_processing_', obj.param.path_res);
                for  id_image=1:1:size(obj.data_img.image, 3)
                    obj.data_img.image(:,:,id_image) = fct_bilateral_filtering(obj.data_img.image(:,:,id_image), 20, .3, 10, 6);
                end
                save_surf_image(obj.data_img.image, '_after_post_processing_', obj.param.path_res);
            end
            % --- get the amplitude corresponding to the scatterers position and
            % store results in cellule -- loop over parameters and sequence
            I = obj.data_img.image(:,:,1);
            max_val = double(max(I(:)));
            % --- select the desired distribution
            if isequal(obj.param.distribution, [1; 0; 0]) % simple interpolation
                corrected_B_mode=(double(I(:,:,1))/max_val).^(1/obj.param.gamma);
            elseif isequal(obj.param.distribution, [0; 1; 0]) % interpolation times normal distribution
                corrected_B_mode=(double(I(:,:,1))/max_val).^(1/obj.param.gamma).*randn(size(obj.data_img.image(:,:,1)));
            elseif isequal(obj.param.distribution, [0; 0; 1]) % interpolation times rayleight distribution
                corrected_B_mode=(double(I(:,:,1))/max_val).^(1/obj.param.gamma).*raylrnd(1,obj.data_img.height,obj.data_img.width)/sqrt(pi/2);
            end
            % --- inverse operation of the log compression
            interpolant=scatteredInterpolant(x_m(:),z_m(:),corrected_B_mode(:), 'linear');
            % --- randomly generate the scatterers position
            rng("shuffle")
            
            if obj.param.random_mode == "QUASI_RANDOM"
                
                p = haltonset(2,'Skip',1e3,'Leap',1e2);
                p = scramble(p,'RR2');
                pts = net(p, fix(nb_scatteres_speckle));
                x_scat_ = pts(:,1) * obj.data_img.width;
                z_scat_ = pts(:,2) * obj.data_img.height;

            elseif obj.param.random_mode == "UNIFORM"
            
                x_scat_=random('unif', 0, obj.data_img.width, fix(nb_scatteres_speckle), 1);
                z_scat_=random('unif', 0, obj.data_img.height, fix(nb_scatteres_speckle), 1);
            
            end         

            % --- get the reflexion coefficient
            RC_scat=interpolant(x_scat_, z_scat_);
            % --- convert x_scat and z_scatt in m
            x_scat=x_scat_*obj.data_img.CF-obj.data_img.width*obj.data_img.CF/2;         % we center x_scatt around 0
            z_scat=z_scat_*obj.data_img.CF;
            % --- store the results
            obj.data_scatt.z_max=z_max+obj.param.shift;
            obj.data_scatt.z_min=z_min;
            obj.data_scatt.x_max=x_max;
            obj.data_scatt.x_min=x_min;
            obj.data_scatt.y_max=0;
            obj.data_scatt.y_min=0;
            obj.data_scatt.x_scatt=x_scat;
            obj.data_scatt.z_scatt=z_scat+obj.param.shift;
            obj.data_scatt.y_scatt=zeros(size(x_scat, 1), 1);
            obj.data_scatt.RC_scatt=RC_scat;
            obj.data_scatt.depth_of_focus=3/4*(z_max)+obj.param.shift;

            obj.data_scatt_moved = obj.data_scatt;
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function save_scatteres(obj)
            % Save scatterers map in png format and in .mat file.
           
            % --- save data                    
            volume = find(obj.data_scatt.y_scatt~=0);
            
            scatt_data_to_save = obj.data_scatt_moved;

            scatt_data_to_save.x_offset_scaling = obj.x_offset_scaling;
            scatt_data_to_save.y_offset_scaling = obj.y_offset_scaling;
            scatt_data_to_save.z_offset_scaling = obj.z_offset_scaling;
            
            scatt_data_to_save.x_offset_shearing = obj.x_offset_shearing;
            scatt_data_to_save.y_offset_shearing = obj.y_offset_shearing;
            scatt_data_to_save.z_offset_shearing = obj.z_offset_shearing;
        
            scatt_data_to_save.x_offset_stretch = obj.x_offset_stretch;
            scatt_data_to_save.y_offset_stretch = obj.y_offset_stretch;
            scatt_data_to_save.z_offset_stretch = obj.z_offset_stretch;
            
            scatt_data_to_save.f_simu = obj.f_simu;
            scatt_data_to_save.theta_max_rot = obj.theta_max_rot;
            scatt_data_to_save.theta_max_shear = obj.theta_max_shear;
            scatt_data_to_save.scaling_coef = obj.scaling_coef;
            scatt_data_to_save.stretch_coef = obj.stretch_coef;
            scatt_data_to_save.time_sample_simu = obj.time_sample_simu;      % the total number of time samples for simulation, but only 3 are used
            scatt_data_to_save.fps = obj.fps;

            if obj.id_seq <10
                str_scat_id = ['_id_sequence_' '00' num2str(obj.id_seq)];
            elseif obj.id_seq <100
                str_scat_id = ['_id_sequence_'  '0' num2str(obj.id_seq)];
            else
                str_scat_id = ['_id_sequence_' num2str(obj.id_seq)];
            end
            

            seg_LI_ = obj.seg_LI;
            seg_MA_ = obj.seg_MA;

            org_dim = linspace(1, ...
                               obj.data_img.width, ...
                               obj.data_img.width)' * obj.data_img.CF;
            org_dim = org_dim - max(org_dim)/2;

            LI_val.seg = interp1(seg_LI_.x_scatt, seg_LI_.z_scatt, org_dim, "linear", 0) / obj.data_img.CF + 1;
            MA_val.seg = interp1(seg_MA_.x_scatt, seg_MA_.z_scatt, org_dim) / obj.data_img.CF + 1; 
            
            I = obj.data_img.image;
            
            for i=1:1:obj.data_img.width                                                                           
                if round(MA_val.seg(i))>0
                    I(round(MA_val.seg(i)), i) = 255;                                           
                end
                if round(LI_val.seg(i))>0   
                    I(round(LI_val.seg(i)), i) = 255; 
                end
            end                                                                                                          

            
            LI_val.seg = LI_val.seg - obj.param.remove_top_region/obj.data_img.CF;  
            MA_val.seg = MA_val.seg - obj.param.remove_top_region/obj.data_img.CF;

            if isempty(volume)
                fct_save_scatterers_2D(obj.data_scatt, obj.param, '');
                fct_save_scatterers_2D(scatt_data_to_save, obj.param, str_scat_id)
                
           
                LI_path=fullfile(obj.param.path_res, 'phantom', 'LI.mat');
                MA_path=fullfile(obj.param.path_res, 'phantom', 'MA.mat');

                save(MA_path, 'MA_val');
                save(LI_path, 'LI_val');        
    
            else
                fct_save_scatterers_3D(obj.data_scatt, obj.param, '');
                fct_save_scatterers_3D(scatt_data_to_save, obj.param, str_scat_id)
           
                LI_path=fullfile(obj.param.path_res, 'phantom', 'LI.mat');
                MA_path=fullfile(obj.param.path_res, 'phantom', 'MA.mat');

                save(MA_path, 'MA_val');
                save(LI_path, 'LI_val');
            end
                    

        end

        % ---------------------------------------------------------------------------------------------------------------------- 
        function get_seg(obj, path_LI, path_MA)
               
            % --- load segmentation
            format_spec = '%f';

            fileID=fopen(path_LI, 'r');
            LI = fscanf(fileID, format_spec); 
            LI = LI(2:2:end);

            fileID=fopen(path_MA, 'r');
            MA = fscanf(fileID, format_spec); 
            MA = MA(2:2:end);
            
            % --- adapt according to offset (image cropping)
            LI = LI(obj.data_img.x_start:end-obj.data_img.x_end) - (obj.data_img.z_start - 1) + 1;
            MA = MA(obj.data_img.x_start:end-obj.data_img.x_end) - (obj.data_img.z_start - 1) + 1;
            
            % --- center aound 0
            LI_x = (0:1:length(LI)-1) * obj.data_img.CF;
            LI_max = max(LI_x);
            LI_x = LI_x - LI_max/2;

            MA_x = (0:1:length(MA)-1) * obj.data_img.CF;
            MA_max = max(MA_x);
            MA_x = MA_x - MA_max/2;
        
            % --- get corresponding non-zeros scatterers
            LI_x_pos = find(LI>1);
            MA_x_pos = find(MA>1);
            
            % --- save values
            obj.seg_LI.x_scatt = LI_x(LI_x_pos)';
            obj.seg_MA.x_scatt = MA_x(MA_x_pos)';

            obj.seg_LI.z_scatt = (LI(LI_x_pos) - 1) * obj.data_img.CF;
            obj.seg_MA.z_scatt = (MA(MA_x_pos) - 1) * obj.data_img.CF;

            obj.seg_LI.y_scatt = zeros(size(obj.seg_LI.z_scatt, 1), 1);
            obj.seg_MA.y_scatt = zeros(size(obj.seg_MA.z_scatt, 1), 1);

            obj.seg_LI.x_min = min(obj.seg_LI.x_scatt);
            obj.seg_LI.x_max = max(obj.seg_LI.x_scatt);
            obj.seg_MA.x_min = min(obj.seg_MA.x_scatt);
            obj.seg_MA.x_max = max(obj.seg_MA.x_scatt);
            
            obj.seg_LI.z_min = min(obj.seg_LI.z_scatt);
            obj.seg_LI.z_max = max(obj.seg_LI.z_scatt);
            obj.seg_MA.z_min = min(obj.seg_MA.z_scatt);
            obj.seg_MA.z_max = max(obj.seg_MA.z_scatt);

            obj.seg_LI.y_max = 0;
            obj.seg_LI.y_min = 0;
            obj.seg_MA.y_max = 0;
            obj.seg_MA.y_min = 0;

            obj.seg_LI_ref = obj.seg_LI;
            obj.seg_MA_ref = obj.seg_MA;
        end
  
        % ----------------------------------------------------------------------------------------------------------------------
        function remove_top_region(obj, dist)
            % Remove scatteres located under dist (in meter).

            idx = find(obj.data_scatt_moved.z_scatt > dist);
            
            obj.data_scatt_moved.z_scatt=obj.data_scatt_moved.z_scatt(idx);
            obj.data_scatt_moved.x_scatt=obj.data_scatt_moved.x_scatt(idx);
            obj.data_scatt_moved.y_scatt=obj.data_scatt_moved.y_scatt(idx);
            obj.data_scatt_moved.RC_scatt=obj.data_scatt_moved.RC_scatt(idx);
            
            % --- modify min and max value
            obj.data_scatt_moved.z_min = min(obj.data_scatt_moved.z_scatt); 
            obj.data_scatt_moved.z_max = max(obj.data_scatt_moved.z_scatt);
            
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function extrusion(obj, extrusion)
            % This function extracts scatterers located in a plan. Then
            % scatteres position are randomly chosen in the corresponding
            % volume.
            
            if extrusion
                
                slice_spacing=obj.param.slice_spacing;
                iter = obj.param.nb_slices;
                nb_scat=length(obj.data_scatt.x_scatt);

                % --- copy of scatterers in 2D plan
                x_scatt = obj.data_scatt.x_scatt;
                z_scatt = obj.data_scatt.z_scatt;
                RC_scatt = obj.data_scatt.RC_scatt;

                for i=1:iter
                    % --- compute y-position 
                    pos = i * slice_spacing;

                    % --- do it twice time for +/- pos
                    y_scatt_1 = ones(nb_scat, 1) * pos;                
                    y_scatt_2 = -y_scatt_1 ;

                    % --- get new values
                    obj.data_scatt.x_scatt =  [obj.data_scatt.x_scatt   ; x_scatt    ; x_scatt  ];
                    obj.data_scatt.y_scatt =  [obj.data_scatt.y_scatt   ; y_scatt_1  ; y_scatt_2];
                    obj.data_scatt.z_scatt =  [obj.data_scatt.z_scatt   ; z_scatt    ; z_scatt  ];
                    obj.data_scatt.RC_scatt = [obj.data_scatt.RC_scatt  ; RC_scatt   ; RC_scatt ];

                end
                
                % --- phantom dimension
                z_min = min(obj.data_scatt.z_scatt);
                z_max = max(obj.data_scatt.z_scatt);
                x_min = min(obj.data_scatt.x_scatt);
                x_max = max(obj.data_scatt.x_scatt);
                y_min = min(obj.data_scatt.y_scatt);
                y_max = max(obj.data_scatt.y_scatt);
                x_img = x_max - x_min;
                z_img = z_max - z_min;
                y_img = y_max - y_min;
                
                % --- interpolation in 3D space
                lambda = obj.param.c / obj.param.fc;
                nb_scat = round(obj.param.scat_density * x_img * y_img * z_img / (lambda^3));
                interpolant = scatteredInterpolant(obj.data_scatt.x_scatt, ...
                                                   obj.data_scatt.z_scatt, ...
                                                   obj.data_scatt.y_scatt, ...
                                                   obj.data_scatt.RC_scatt);
                
                
                if obj.param.random_mode == "QUASI_RANDOM"
                    p = haltonset(3,'Skip', 1e3,'Leap', 1e2);
                    p = scramble(p, 'RR2');
                    pts = net(p, fix(nb_scat));
                    x_scat = pts(:,1) * (x_max - x_min) + x_min;
                    y_scat = pts(:,2) * (y_max - y_min) + y_min;
                    z_scat = pts(:,3) * (z_max - z_min) + z_min;
                elseif obj.param.random_mode == "UNIFORM"
                    y_scat = random('unif', y_min, y_max, nb_scat, 1);
                    x_scat= random('unif', x_min, x_max, nb_scat, 1); 
                    z_scat= random('unif', z_min, z_max, nb_scat, 1);
                end

                RC_scatt = interpolant(x_scat, z_scat, y_scat);
                % --- update
                obj.data_scatt.x_scatt = x_scat;
                obj.data_scatt.y_scatt = y_scat;
                obj.data_scatt.z_scatt = z_scat;
                obj.data_scatt.RC_scatt = RC_scatt;
                obj.data_scatt.y_min = y_min;
                obj.data_scatt.y_max = y_max;
                
            else
                nb_scat=length(obj.data_scatt.x_scatt);
                obj.data_scatt.y_scatt = zeros(nb_scat, 1);
            end
            
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function init_position(obj)
            % Initialisation of the following parameters:
            % offset_rot -> x, y and z position
            % x_offset_scaling -> x, y and z position
            % x_offset_shearing -> x, y and z position
            % x_offset_stretching -> x, y and z position
            % cardiac_cycle_bpm
            % theta_max_rot -> magnitude
            % scaling_coef -> magnitude
            % scaling_stretching -> magnitude
            % scaling_shearing -> magnitude
            % fps -> frame per second

            rng("shuffle")

            % --- get three random x_offset/z_offset/y_offset
            x_min = obj.data_scatt.x_min;
            x_max = obj.data_scatt.x_max;
            z_min = obj.data_scatt.z_min;
            z_max = obj.data_scatt.z_max;
            y_min = obj.data_scatt.y_min;
            y_max = obj.data_scatt.y_max;
            
            % --- offset for rotation
            obj.x_offset_rot = x_min + (x_max - x_min) * rand(1,1);
            obj.z_offset_rot = z_min + (z_max - z_min) * rand(1,1);
            obj.y_offset_rot = y_min + (y_max - y_min) * rand(1,1);
            % --- offset for scaling
            obj.x_offset_scaling = x_min + (x_max - x_min) * rand(1,1);
            obj.z_offset_scaling = z_min + (z_max - z_min) * rand(1,1);
            obj.y_offset_scaling = y_min + (y_max - y_min) * rand(1,1);
            % --- offset for shearing
            obj.x_offset_shearing = x_min + (x_max - x_min) * rand(1,1);
            obj.z_offset_shearing = z_min + (z_max - z_min) * rand(1,1);
            obj.y_offset_shearing = y_min + (y_max - y_min) * rand(1,1);
            % --- offset for stretch
            obj.x_offset_stretch = x_min + (x_max - x_min) * rand(1,1);
            obj.z_offset_stretch = z_min + (z_max - z_min) * rand(1,1);
            obj.y_offset_stretch = y_min + (y_max - y_min) * rand(1,1);
            
            % --- get random coefficient 
            % -- CARDIAC CYCLE
            a = obj.param.cardiac_cycle_bpm(1);
            b = obj.param.cardiac_cycle_bpm(2);
            obj.f_simu = a + (b - a) * rand(1,1);
            % -- THETA ROTATION
            a = obj.param.theta_max_rot(1);
            b = obj.param.theta_max_rot(2);
            obj.theta_max_rot = a + (b - a) * rand(1,1);
            % -- THETA SHEARING
            a = obj.param.theta_max_shear(1);
            b = obj.param.theta_max_shear(2);
            obj.theta_max_shear = a + (b - a) * rand(1,1);
            % -- SCALING COEFFICIENT
            a = obj.param.scaling_coef(1);
            b = obj.param.scaling_coef(2);
            obj.scaling_coef = a + (b - a) * rand(1,1);
            % -- STRETCHING
            a = obj.param.stretch_coef(1);
            b = obj.param.stretch_coef(2);
            obj.stretch_coef = a + (b - a) * rand(1,1);
            % -- FRAME PER SECOND
            a = obj.param.fps(1);
            b = obj.param.fps(2);
            obj.fps = a + (b - a) * rand(1,1);

            % --- get index to initialize the first sequence
            T=1/obj.f_simu;
            dt=T/obj.fps;
            obj.time_sample_simu = 0:dt:4*T; 
            obj.id_seq = 1;
        end

        % ----------------------------------------------------------------------------------------------------------------------
        function incr_id_sequence(obj)
            % Increment id seq after applying motion to the scatteres map.

            obj.id_seq = obj.id_seq+1;
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function animate_scatteres(obj)
            % Apply motion to scatteres map

            obj.data_scatt_moved = obj.data_scatt;
            
            % --- offset
            offset_rot = [obj.x_offset_rot, obj.y_offset_rot, obj.z_offset_rot];
            offset_shearing = [obj.x_offset_shearing, obj.y_offset_shearing, obj.z_offset_shearing];
            offset_scaling = [obj.x_offset_scaling, obj.y_offset_scaling, obj.z_offset_scaling];
            offset_stretch = [obj.x_offset_stretch, obj.y_offset_stretch, obj.z_offset_stretch];

            offset.shearing = offset_shearing;
            offset.rot = offset_rot;
            offset.scaling= offset_scaling;
            offset.stretch = offset_stretch;
            
            % --- param simu
            param_simu.theta_rot = obj.theta_max_rot;
            param_simu.theta_shearing = obj.theta_max_shear;
            param_simu.scaling_coef = obj.scaling_coef;
            param_simu.stretch_coef = obj.stretch_coef;
            param_simu.f_simu = obj.f_simu;
            param_simu.id_seq = obj.id_seq;
            param_simu.time_sample_simu = obj.time_sample_simu;

            % --- gaussian
            gaussian.x = obj.gaussian_x;
            gaussian.y = obj.gaussian_y; 
            gaussian.z = obj.gaussian_z;
             
            obj.data_scatt_moved = add_movement(obj.data_scatt_moved,offset, param_simu, gaussian); % scatteres map
            obj.seg_LI = add_movement(obj.seg_LI_ref, offset, param_simu, gaussian);                % segmentation (LI)
            obj.seg_MA = add_movement(obj.seg_MA_ref, offset, param_simu, gaussian);                % segmentation (MA)
            
        end
        
         % ----------------------------------------------------------------------------------------------------------------------
        function update_parameters(obj, param)
            % Load parameters and store them in a structure.
            
            list_files_param=fct_list_mat_files(param, 'parameters');

            for id_param=1:1:size(list_files_param, 2)
                obj.param=fct_load_param(fullfile(param, 'parameters', list_files_param{id_param}));
            end
        end

        % ----------------------------------------------------------------------------------------------------------------------
        function clear_moved_image(obj)
            % Remove out-of-of scatteres

            z_max = obj.data_scatt.z_max;
            z_min = obj.data_scatt.z_min;
            
            x_max = obj.data_scatt.x_max;
            x_min = obj.data_scatt.x_min;
            
            phantom = obj.data_scatt_moved;
            
            % --- remove out of box scatterers (<z_max)
            idx = find(phantom.z_scatt < z_max);
            phantom.z_scatt=phantom.z_scatt(idx);
            phantom.x_scatt=phantom.x_scatt(idx);
            phantom.y_scatt=phantom.y_scatt(idx);
            phantom.RC_scatt=phantom.RC_scatt(idx);

            % --- remove out of box scatterers (>z_min)
            idx = find(phantom.z_scatt > z_min);
            phantom.z_scatt=phantom.z_scatt(idx);
            phantom.x_scatt=phantom.x_scatt(idx);
            phantom.y_scatt=phantom.y_scatt(idx);
            phantom.RC_scatt=phantom.RC_scatt(idx);

            % --- remove out of box scatterers (>x_min)
            idx = find(phantom.x_scatt > x_min);
            phantom.z_scatt=phantom.z_scatt(idx);
            phantom.x_scatt=phantom.x_scatt(idx);
            phantom.y_scatt=phantom.y_scatt(idx);
            phantom.RC_scatt=phantom.RC_scatt(idx);

            % --- remove out of box scatterers (<x_max)
            idx = find(phantom.x_scatt < x_max);
            phantom.z_scatt=phantom.z_scatt(idx);
            phantom.x_scatt=phantom.x_scatt(idx);
            phantom.y_scatt=phantom.y_scatt(idx);
            phantom.RC_scatt=phantom.RC_scatt(idx);
            
            obj.data_scatt_moved = phantom;
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function init_scatt_reference(obj)
            % Get the map of the scatterers that correspond to 
            % the original image (each scatterer is located in 
            % the center of each pixel).

            % --- get dimension of the image
            width_img = obj.data_img.width;
            height_img = obj.data_img.height;
            
            x = width_img * obj.data_img.CF;
            z = height_img * obj.data_img.CF;
            x_real = linspace(-x/2, x/2, width_img);
            z_real = linspace(0, z, height_img)';
            
            % --- generate scattereres
            [X, Z] = meshgrid(x_real, z_real);
            x_scatt_ref = X(:);
            z_scatt_ref = Z(:);
            y_scatt_ref = 0 * X(:);    
            
            obj.scatt_pos_ref{1}.x_scatt = x_scatt_ref;
            obj.scatt_pos_ref{1}.z_scatt = z_scatt_ref;
            obj.scatt_pos_ref{1}.y_scatt = y_scatt_ref;

            obj.scatt_pos_ref{1}.x_min = min(x_scatt_ref);
            obj.scatt_pos_ref{1}.y_min = min(y_scatt_ref);
            obj.scatt_pos_ref{1}.z_min = min(z_scatt_ref);

            obj.scatt_pos_ref{1}.x_max = max(x_scatt_ref);
            obj.scatt_pos_ref{1}.y_max = max(y_scatt_ref);
            obj.scatt_pos_ref{1}.z_max = max(z_scatt_ref);

%             fct_save_scatt_ref(obj.scatt_pos_ref{1}, obj.data_img.height, fullfile(obj.param.path_res, 'phantom', 'scatt_ref_org'));

        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function create_OF_GT(obj, id_img)
            % Add motion to the grid which corresponds to the  original
            % image.

            scatt_ref_moved = obj.scatt_pos_ref{1};
            
            % --- offset
            offset_rot = [obj.x_offset_rot, obj.y_offset_rot, obj.z_offset_rot];
            offset_shearing = [obj.x_offset_shearing, obj.y_offset_shearing, obj.z_offset_shearing];
            offset_scaling = [obj.x_offset_scaling, obj.y_offset_scaling, obj.z_offset_scaling];
            offset_stretch = [obj.x_offset_stretch, obj.y_offset_stretch, obj.z_offset_stretch];

            offset.shearing = offset_shearing;
            offset.rot = offset_rot;
            offset.scaling= offset_scaling;
            offset.stretch = offset_stretch;
            
            param_simu.theta_rot = obj.theta_max_rot;
            param_simu.theta_shearing = obj.theta_max_shear;
            param_simu.scaling_coef = obj.scaling_coef;
            param_simu.stretch_coef = obj.stretch_coef;
            param_simu.f_simu = obj.f_simu;
            param_simu.id_seq = obj.id_seq;
            param_simu.time_sample_simu = obj.time_sample_simu;

            % --- gaussian
            gaussian.x = obj.gaussian_x;
            gaussian.y = obj.gaussian_y; 
            gaussian.z = obj.gaussian_z;
             
            obj.scatt_pos_ref{2} = scatt_ref_moved;
            scatt_ref_moved = add_movement(scatt_ref_moved, offset, param_simu, gaussian);
            obj.scatt_pos_ref{3} = scatt_ref_moved;
            
            % --- compute flow
            if id_img > 1
                
                Pos_0=[obj.scatt_pos_ref{2}.x_scatt obj.scatt_pos_ref{2}.y_scatt obj.scatt_pos_ref{2}.z_scatt];
                Pos_1=[obj.scatt_pos_ref{3}.x_scatt obj.scatt_pos_ref{3}.y_scatt obj.scatt_pos_ref{3}.z_scatt];
                diff=Pos_1-Pos_0;

                % --- flow in meter
                flow = reshape(diff,[obj.data_img.height obj.data_img.width 3]); 
                pixel_size = obj.data_img.CF;

                % --- convert the flow in pixels displacment
                flow = flow/pixel_size;
%                 % can't be used on VIP platform
%                 fct_save_scatt_ref(obj.scatt_pos_ref{1}, obj.data_img.height, ...
%                                    fullfile(obj.param.path_res, 'phantom', ['scatt_pos_org' num2str(obj.id_seq-1)]));
%                 
%                 fct_save_scatt_ref(obj.scatt_pos_ref{2}, obj.data_img.height, ...
%                                    fullfile(obj.param.path_res, 'phantom', ['scatt_pos_id_' num2str(obj.id_seq-1)]));
%                 
%                 fct_save_scatt_ref(obj.scatt_pos_ref{3}, obj.data_img.height, ...
%                                    fullfile(obj.param.path_res, 'phantom', ['scatt_pos_id_' num2str(obj.id_seq)]));
                
                fct_save_OF_GT(flow, fullfile(obj.param.path_res, 'phantom', ['OF_' num2str(obj.id_seq-1) '_' num2str(obj.id_seq)]));
%                 % can't bu used on VIP platform
%                 fct_save_flow(flow, fullfile(obj.param.path_res, 'phantom', ['optical_pos_id_', num2str(obj.id_seq)]));
            
            else
%                 % can't be used on VIP platform
%                 fct_save_scatt_ref(obj.scatt_pos_ref{3}, obj.data_img.height, ...
%                                    fullfile(obj.param.path_res, 'phantom', ['scatt_pos_id_' num2str(obj.id_seq)]));
            end
            
%             obj.scatt_pos_ref{2} = obj.scatt_pos_ref{3};
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function save_image(obj)
            % Save original image information map and the original image .png.
            
            image = obj.data_img; 
            parameters = obj.param;
            % --- save variables in .mat format
            path_res=fullfile(parameters.path_res, 'phantom/image_information.mat');
            save(path_res, 'image');
            
        end
        
        % ----------------------------------------------------------------------------------------------------------------------
        function get_gaussian_parameters(obj, n)
            % we combine n gaussians with different parameters

            rng("shuffle")
            
            % --- define interval
            x_min = obj.scatt_pos_ref{1}.x_min;
            x_max = obj.scatt_pos_ref{1}.x_max;
            y_max = obj.scatt_pos_ref{1}.y_min;
            y_min = obj.scatt_pos_ref{1}.y_max;
            z_max = obj.scatt_pos_ref{1}.z_min;
            z_min = obj.scatt_pos_ref{1}.z_max;
            
            % --- define paremeters
            sigma_x = [(x_max-x_min)/100 (x_max-x_min)/10];
            sigma_y = [(y_max-y_min)/100 (y_max-y_min)/10];
            sigma_z = [(z_max-z_min)/100 (z_max-z_min)/10];
            
            % amp = [0 0.005];
            amp = [0 0];
            
            mu_x = [x_min x_max];
            mu_y = [y_min y_max];
            mu_z = [z_min z_max];
                        
            for id=1:1:n
                
                % --- x-axis
                mu_x_ = mu_x(1) + (mu_x(2) - mu_x(1)) * rand(1, 1);
                sigma_x_ = sigma_x(1) + (sigma_x(2) - sigma_x(1)) * rand(1, 1);
                amp_x_ = amp(1) + (amp(2) - amp(1)) * rand(1, 1);

                obj.gaussian_x{id}.mu_x = mu_x_;
                obj.gaussian_x{id}.sigma_x = sigma_x_;
                obj.gaussian_x{id}.amp_x = amp_x_;
                

                % --- y-axis
                mu_y_ = mu_y(1) + (mu_y(2) - mu_y(1)) * rand(1, 1);
                sigma_y_ = sigma_y(1) + (sigma_y(2) - sigma_y(1)) * rand(1, 1);
                amp_y_ = amp(1) + (amp(2) - amp(1)) * rand(1, 1);

                obj.gaussian_y{id}.mu_y = mu_y_;
                obj.gaussian_y{id}.sigma_y = sigma_y_;
                obj.gaussian_y{id}.amp_y = amp_y_;

                % --- z-axis
                mu_z_ = mu_z(1) + (mu_z(2) - mu_z(1)) * rand(1, 1);
                sigma_z_ = sigma_z(1) + (sigma_z(2) - sigma_z(1)) * rand(1, 1);
                amp_z_ = amp(1) + (amp(2) - amp(1)) * rand(1, 1);

                obj.gaussian_z{id}.mu_z = mu_z_;
                obj.gaussian_z{id}.sigma_z = sigma_z_;
                obj.gaussian_z{id}.amp_z = amp_z_;
            end

        end

        % ----------------------------------------------------------------------------------------------------------------------
        function phantom_tmp(obj)
            
            k=5;

            z_pos=linspace(obj.data_scatt.z_max*0.1, obj.data_scatt.z_max, k);
            x_pos=linspace(obj.data_scatt.x_min,     obj.data_scatt.x_max, k);

            obj.data_scatt.y_scatt=[];
            obj.data_scatt.x_scatt=[];
            obj.data_scatt.z_scatt=[];
            obj.data_scatt.RC_scatt=[];

            for i=1:1:k
                for j=1:k
                    obj.data_scatt.y_scatt=[obj.data_scatt.y_scatt; 0];
                    obj.data_scatt.z_scatt=[obj.data_scatt.z_scatt; z_pos(j)];
                    obj.data_scatt.x_scatt=[obj.data_scatt.x_scatt; x_pos(i)];
                    obj.data_scatt.RC_scatt=[obj.data_scatt.RC_scatt; 1];
                end
            end

            obj.data_scatt.z_scatt=obj.data_scatt.z_scatt+obj.param.shift;
            obj.data_scatt.depth_of_focus=max(obj.data_scatt.z_scatt);
            obj.data_scatt_moved = obj.data_scatt;

        end
        
    end
end

% ------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------
function [struct_image]=load_JPEG(path_img)   
    % Loads tiff image

    % --- read the image
    image=imread(path_img);
    if size(image, 3) == 3
        image=rgb2gray(image); % we convert in grayscale
    end
    
    % --- we define the size of the pixels in a random way according to the average size of the pixels of US 
    % probe around 7MHz
    rng('shuffle')
    CF_max = 100e-6;
    CF_min = 30e-6;
    
    CF = CF_min + (CF_max - CF_min) * rand(1,1);

    % --- fill obj
    struct_image.CF=CF;
    struct_image.width=size(image, 2);
    struct_image.height=size(image, 1);
    struct_image.image=image;

end

% ------------------------------------------------------------------------------------------------------------------------------
function [struct_image]=load_tiff(path_img, roi)   
    % Loads tiff image

    % --- read the image
    image=imread(path_img);
    if size(image, 3) == 3
        image=rgb2gray(image); % we convert in grayscale
    end

    
    x_start=roi(1);
    x_end=roi(2);
    y_start=roi(3);
    y_end=roi(4);

    image=image(y_start:end-y_end, x_start:end-x_end); % we only consider the ROI, change later

    % --- load the size of the pixels
    tmp_=split(path_img, '/');
    str_=fct_build_path(tmp_, 2);
    patient_id=tmp_{end};
    patient_id=split(patient_id, '.');
    patient_id=patient_id{1};
    path_to_cf=fullfile(str_, 'CF', strcat(patient_id, '_CF.txt'));
    format_spec = '%f';
    fileID=fopen(path_to_cf, 'r');
    CF=fscanf(fileID,format_spec)*1e-3; % pixel size in m
    
    % --- fill obj
    struct_image.CF=CF;
    struct_image.width=size(image, 2);
    struct_image.height=size(image, 1);
    struct_image.image=image;
    struct_image.x_start = x_start;
    struct_image.x_end = x_end;
    struct_image.z_start = y_start;
    struct_image.z_end = y_end;

end

% ------------------------------------------------------------------------------------------------------------------------------
function [struct_image]=load_dicom(path_img, sequence) 
    % Load DICOM sequence.

    % --- probe info
    info=dicominfo(path_img);
    rect = [info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0 + 80,...
        info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0 + 50,... # 50
        info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1 - 185,...
        info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1 - 110];

    % --- we load the information
    if sequence
        image_=dicomread(path_img);
        for i=1:1:size(image_, 4)
            image(:,:,i)=rgb2gray(imcrop(image_(:,:,:,i), rect));
        end
    else
        image=dicomread(path_img, 'frame', 1); % we load the first frame of the sequence
        image=rgb2gray(imcrop(image,rect));
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
function [phantom_out] = add_rotation(phantom_in, theta_max, id, time, f)
    
    theta_sup_rot = (theta_max*pi)/180;
    motion_rot = @(t) theta_sup_rot * sin(2 * pi * f * t);


    % --- AFFINE TRANSFORMATION
    % -- rotation transformation
    rot_transf=@(x) [cos(x), sin(x),       0, 0; 
                     -sin(x), cos(x),      0, 0; 
                     0,       0,           1, 0;
                     0,       0,           0, 1];

    % --- INITIALIZATION       

    phantom_out = phantom_in;
    rotation = arrayfun( @(x) (affine3d(rot_transf(x))), motion_rot(time));
    projection = rotation(id).transformPointsForward(cat(2, phantom_in.x_scatt,...
                                                            phantom_in.z_scatt,...
                                                            phantom_in.y_scatt));
  
%     projection = rotation(id).transformPointsForward([phantom_in.x_scatt,phantom_in.z_scatt,phantom_in.y_scatt]);
    % --- provide new attributes
    phantom_out.x_scatt = projection(:,1);
    phantom_out.z_scatt = projection(:,2);
    phantom_out.y_scatt = projection(:,3);

end

% ------------------------------------------------------------------------------------------------------------------------------
function [phantom_out] = add_shearing(phantom_in, theta_max, id, time, f)
    
    theta_sup_shear = (theta_max*pi)/180;            
    motion_shear = @(t) theta_sup_shear * sin(2 * pi * f * t);


    % --- AFFINE TRANSFORMATION
    % -- rotation transformation
    shear_transf=@(x) [1,      0,   0,   0; 
                       tan(x), 1,   0,   0; 
                       0,      0,   1,   0
                       0,      0,   0,   1];

    % --- INITIALIZATION       

    phantom_out = phantom_in;
    shearing = arrayfun( @(x) (affine3d(shear_transf(x))), motion_shear(time));
    projection = shearing(id).transformPointsForward(cat(2, phantom_in.x_scatt,...
                                                            phantom_in.z_scatt,...
                                                            phantom_in.y_scatt));
    % --- provide new attributes
    phantom_out.x_scatt = projection(:,1);
    phantom_out.z_scatt = projection(:,2);
    phantom_out.y_scatt = projection(:,3);
    
end

% ------------------------------------------------------------------------------------------------------------------------------
function [phantom_out] = add_scaling(phantom_in, coef_scaling, id, time, f)
    
    motion_scale=@(t) coef_scaling/100 * sin(2 * pi * f * t);

    % --- AFFINE TRANSFORMATION
    % -- rotation transformation
    scale_transf=@(x) [1, 0, 0, 0; 
                       0, 1+x, 0, 0; 
                       0, 0, 1+x, 0; 
                       0, 0, 0, 1];

    % --- INITIALIZATION       

    phantom_out = phantom_in;
    scaling = arrayfun( @(x) (affine3d(scale_transf(x))), motion_scale(time));
    projection = scaling(id).transformPointsForward(cat(2, phantom_in.x_scatt,...
                                                           phantom_in.z_scatt,...
                                                           phantom_in.y_scatt));
    % --- provide new attributes
    phantom_out.x_scatt = projection(:,1);
    phantom_out.z_scatt = projection(:,2);
    phantom_out.y_scatt = projection(:,3);
    
end

% ------------------------------------------------------------------------------------------------------------------------------
function [phantom_out] = add_stretch(phantom_in, coef_scaling, id, time, f)
    
    motion_stretch = @(t) coef_scaling/100 * sin(2 * pi * f * t);

    % --- AFFINE TRANSFORMATION
    % -- rotation transformation
    transl_transf=@(x) [1 + x, 0, 0, 0; 
                        0,     1, 0, 0; 
                        0,     0, 1, 0; 
                        0,     0, 0, 1];

    % --- INITIALIZATION       

    phantom_out = phantom_in;
    stretch = arrayfun( @(x) (affine3d(transl_transf(x))), motion_stretch(time));
    projection = stretch(id).transformPointsForward(cat(2, phantom_in.x_scatt,...
                                                            phantom_in.z_scatt,...
                                                            phantom_in.y_scatt));
    % --- provide new attributes
    phantom_out.x_scatt = projection(:,1);
    phantom_out.z_scatt = projection(:,2);
    phantom_out.y_scatt = projection(:,3);
end

% ------------------------------------------------------------------------------------------------------------------------------
function [phantom] = move_origin(phantom, x_off, y_off, z_off)
    % Change origine before applying affine transformation.

    phantom.x_min = phantom.x_min - x_off;
    phantom.x_max = phantom.x_max - x_off;

    phantom.y_min = phantom.y_min - y_off;
    phantom.y_max = phantom.y_max- y_off;
    
    phantom.z_min = phantom.z_min - z_off;
    phantom.z_max = phantom.z_max- z_off;
    
    phantom.x_scatt = phantom.x_scatt- x_off;
    phantom.y_scatt = phantom.y_scatt - y_off;
    phantom.z_scatt = phantom.z_scatt - z_off;
    
end

% ------------------------------------------------------------------------------------------------------------------------------
function [scatt] = add_movement(scatt, offset, param, gaussian)
    % Add motion: rotation, shearing, scaling, stretching, gaussian noise

    % --- ROTATION
    scatt = move_origin(scatt, offset.rot(1), offset.rot(2), offset.rot(3));
    scatt = add_rotation(scatt, param.theta_rot, param.id_seq, param.time_sample_simu, param.f_simu);
    scatt = move_origin(scatt, -offset.rot(1), -offset.rot(2), -offset.rot(3));
    
    % --- SHEARING
    scatt = move_origin(scatt, offset.shearing(1), offset.shearing(2), offset.shearing(3));
    scatt = add_shearing(scatt, param.theta_shearing, param.id_seq, param.time_sample_simu, param.f_simu);
    scatt = move_origin(scatt, -offset.shearing(1), -offset.shearing(2), -offset.shearing(3));
     
    % --- SCALING
    scatt = move_origin(scatt, offset.scaling(1), offset.scaling(2), offset.scaling(3));
    scatt = add_scaling(scatt, param.scaling_coef, param.id_seq, param.time_sample_simu, param.f_simu);
    scatt = move_origin(scatt, -offset.scaling(1), -offset.scaling(2), -offset.scaling(3));

    % --- STRETCHING
    scatt = move_origin(scatt, offset.stretch(1), offset.stretch(2), offset.stretch(3));
    scatt = add_stretch(scatt, param.stretch_coef, param.id_seq, param.time_sample_simu, param.f_simu);
    scatt = move_origin(scatt, -offset.stretch(1), -offset.stretch(2), -offset.stretch(3));
    
%     % --- ADD GAUSSIAN
%     nb_gaussian = length(gaussian.x);
%     gauss = @(x,mu,sig,amp) amp*exp(-(((x-mu).^2)/(2*sig.^2)));
%     
%     for id=1:1:nb_gaussian
%         if mod(id, 2)
%             scatt.x_scatt = scatt.x_scatt + scatt.x_scatt .* gauss(scatt.x_scatt, gaussian.x{id}.mu_x, gaussian.x{id}.sigma_x, gaussian.x{id}.amp_x);
%             scatt.y_scatt = scatt.y_scatt + 3*scatt.y_scatt .* gauss(scatt.y_scatt, gaussian.y{id}.mu_y, gaussian.y{id}.sigma_y, gaussian.y{id}.amp_y);
%             scatt.z_scatt = scatt.z_scatt + scatt.z_scatt .* gauss(scatt.z_scatt, gaussian.z{id}.mu_z, gaussian.z{id}.sigma_z, gaussian.z{id}.amp_z);
%         else
%             scatt.x_scatt = scatt.x_scatt - scatt.x_scatt .* gauss(scatt.x_scatt, gaussian.x{id}.mu_x, gaussian.x{id}.sigma_x, gaussian.x{id}.amp_x);
%             scatt.y_scatt = scatt.y_scatt - 3*scatt.y_scatt .* gauss(scatt.y_scatt, gaussian.y{id}.mu_y, gaussian.y{id}.sigma_y, gaussian.y{id}.amp_y);
%             scatt.z_scatt = scatt.z_scatt - scatt.z_scatt .* gauss(scatt.z_scatt, gaussian.z{id}.mu_z, gaussian.z{id}.sigma_z, gaussian.z{id}.amp_z);
%         end
%     end

end
