classdef parametersHandler < handle 
   
    properties
        param
    end
    
    methods

        % ------------------------------------------------------------------
        function obj=parametersHandler(jname)
            % Constructor
            
            % --- LOAD JSON
            str = fileread(jname);
            obj.param = jsondecode(str);
            
            % --- RELATIVE TO RESULTS
            obj.param.path_data='';
            obj.param.path_res='';   
            obj.param.phantom_name='';

        
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
            path_to_save = fullfile(obj.param.path_res, 'parameters', 'parameters.mat');
            p = obj.param;
            save(path_to_save, 'p');
            
            path_to_save = fullfile(obj.param.path_res, 'parameters', 'parameters.json');
            names = fieldnames(obj.param);
            
            str = jsonencode(p);           
            % add a return character after all commas:
            new_string = strrep(str, ',', ',\n');
            % add a return character after curly brackets:
            new_string = strrep(new_string, '{', '{\n');

            % Write the string to file
            fid = fopen(path_to_save, 'w');
            fprintf(fid, new_string); 
            fclose(fid);
            

        end
        % ------------------------------------------------------------------
    end
    
end

