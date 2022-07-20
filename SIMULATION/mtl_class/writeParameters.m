classdef writeParameters < handle 
   
    properties
        param
    end
    
    methods

        % ------------------------------------------------------------------
        function obj=writeParameters()
            % Constructor
            
            obj.param.path_data='';
            % --- RELATIVE TO RESULTS
            obj.param.path_res='';    % path to save the result
            obj.param.phantom_name='';

            % --- ENTER SIMUS OR FIELD
            obj.param.soft='';

            % --- RELATIVE TO THE SIMULATION
            obj.param.c 			           = 1540;             % sound celerity
            obj.param.fc 			          = 7.5e6;            % central frequency of the probe
            obj.param.scat_density       = 6;			    % number of scatterers per resolution cell
            obj.param.probe_name 	      = 'L12-3v';		    % name of the probe
            obj.param.range_DB 		      = 50;               % attenuation in dB
            obj.param.bandwidth 		     = 93;               % bandewidth (not used)
            obj.param.fnumber 			     = 1;			    % fnumber (used for apodization (emite en receive apod)
            obj.param.Nactive 			     = 65;               % active number for apodization
            obj.param.Nelements 		     = 192;              % numbe rof element of th eprobe
            obj.param.fsCoef 			      = 6;                % sampling frequency -> fs= sCoef*fc
            obj.param.shift 			       = 0;                % shift scatterer to start at x mm instead of 0
            obj.param.gamma 			       = 0.3;              % gamma correction
            obj.param.TGC 			         = true;             % time gain compensation
            obj.param.distribution 		  = [0 0 1];          % [random_point normal rayleight] -> example for rayleight distribution [0 0 1]
            obj.param.imadjust_vals 	   = [0.03 0.75];      % imagejust coefficient for IQ post processing
            obj.param.dynamic_focusing 	= 1;                % dynamic focus using field only
            obj.param.sequence 			    = false;            % create phantom for all images in the sequence
            obj.param.preprocessing 	   = false;            % preprocess bmode image before extracting scatterers -> here we apply bilateral filtering
            obj.param.mode 			        = [0 1];            % [1 0] -> scanline based ; [0 1] -> synthetic aperture
            obj.param.slice_spacing 	   = 0.5e-4;             % space between slice in y direction
            obj.param.nb_slices 		     = 5;                % total number of slice (has to be an even number)
            obj.param.compensation_time  = -1;               % compensation time
            obj.param.remove_top_region  = 1e-3;             % top region removal in m
            obj.param.random_mode        = 'QUASI_RANDOM';   % 'QUASI_RANDOM', 'UNIFORM'

            % --- RELATIVE TO MOVEMENT
            obj.param.cardiac_cycle_bpm = [50 90];         	% 1 since it is ~60 cardiac cycles per minute
            obj.param.theta_max_rot 	  = [0.1 2];          % maximal rotation movement
            obj.param.theta_max_shear 	= [1 10];          	% maximal rotation movement
            obj.param.scaling_coef 		 = [0.1 5];          % scaling coefficient in %
            obj.param.fps 			        = [50 90];          % number of frame per second
            obj.param.stretch_coef 		 = [1 3];            % stretch coefficient
        
        end
        
        
        % ------------------------------------------------------------------
        % ----------------------- SET PARAMETERS ---------------------------
        % ------------------------------------------------------------------
        
        function set_pres(obj, path_res)
           obj.param.path_res = path_res;
        end
        % ------------------------------------------------------------------
        
        function set_path_data(obj, path_data)
           obj.param.path_data=path_data;
        end
        % ------------------------------------------------------------------
        
        function set_phantom_name(obj, phantom_name)
           obj.param.phantom_name=phantom_name;
        end
        % ------------------------------------------------------------------
        
        function set_software(obj, sofware)
           obj.param.soft = sofware;
        end
        % ------------------------------------------------------------------
        
        function set_Nactive(obj, Nactive)
            obj.param.Nactive = Nactive;
        end   
        % ------------------------------------------------------------------
        
        function set_Nelements(obj, Nelements)
            obj.param.Nelements = Nelements;
        end
        % ------------------------------------------------------------------
        
        function set_scatteres_density(obj, scat_density)
            obj.param.scat_density = scat_density;
        end
        % ------------------------------------------------------------------
        
        function set_acquisition_mode(obj, mode)
            
           switch mode
               case 'scanline_based'
                   obj.param.mode=[1 0];
               case 'synthetic_aperture'
                   obj.param.mode=[0 1];
               otherwise
                   disp('Problem with chosen mode.')
           end
           
        end
        % ------------------------------------------------------------------
        
        function set_nb_slice(obj, nb_slices)
            obj.param.nb_slices = nb_slices;
        end
        % ------------------------------------------------------------------
        
        function set_slice_spacing(obj, slice_spacing)
            obj.param.slice_spacing = slice_spacing;
        end
        % ------------------------------------------------------------------
        
        function set_shift(obj, shift)
            obj.param.shift = shift;
        end
        % ------------------------------------------------------------------
        
        function set_dynamic_focusing(obj, da)
            obj.param.dynamic_focusing = da;
        end
        % ------------------------------------------------------------------
        
        function set_compensation_time(obj, compensation_time)
            obj.param.compensation_time = compensation_time;
        end
        
        % ------------------------------------------------------------------
        % ------------------------- UTILS ----------------------------------
        % ------------------------------------------------------------------
        
        function create_directory(obj)
            % Create directory to store results.
            
            dir_=split(obj.param.path_res, '/');
            path_='/';
            for id=2:1:size(dir_, 1)
                if not(isfolder(fullfile(path_, dir_{id})))
                    mkdir(fullfile(path_, dir_{id}));
                    path_=strcat(path_, dir_{id}, '/');
                else
                    path_=strcat(path_, dir_{id}, '/');
                end
            end
            
            
            % --- we create the directory
            necessary_dir{1}='parameters';
            necessary_dir{2}='phantom';
            necessary_dir{3}='raw_data';
            necessary_dir{4}='raw_data/raw_';
            necessary_dir{5}='raw_data/exec_';
            necessary_dir{6}='bmode_result';
            
            for k=1:1:size(necessary_dir, 2)    
                path=fullfile(obj.param.path_res, necessary_dir{k});
                if not(isfolder(path))
                    mkdir(path);
                end 
            end
        end
        % ------------------------------------------------------------------
        
        function save(obj)
            % Save parameters.
            
            % --- save parmaters in .mat format
            path_to_save=fullfile(obj.param.path_res, 'parameters', 'parameters.mat');
            p=obj.param;
            save(path_to_save, 'p');
            
            path_to_save=fullfile(obj.param.path_res, 'parameters', 'parameters.txt');
            names = fieldnames(obj.param);
            
            % --- save parmaters in .txt format
            fid = fopen(path_to_save, 'w');
            for i=1:1:length(names)
                str_ = ['obj.param.' names{i} '=' num2str(obj.param.(names{i})) ';'];
                fprintf(fid, [str_  '\n']);
            end
            fclose(fid);

        end
        % ------------------------------------------------------------------
    end
    
end

