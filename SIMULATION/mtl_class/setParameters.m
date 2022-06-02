classdef setParameters < handle 
   
    properties
        
        parameters;
      
    end
    
    methods
        % ------------------------------------------------------------------
        function obj=setParameters(parameters, name_m_file)
            % Constructor
            
            obj.parameters=parameters;
            obj.create_directory();
            obj.make_copy(name_m_file);
            obj.save();
        end
        % ------------------------------------------------------------------
        function make_copy(obj, name_m_file)
            % Make a copy of parameters in order to track the parameters
            % used for simulation.
            path_to_copy=fullfile(obj.parameters.path_res, 'parameters');
            copyfile(name_m_file, fullfile(path_to_copy, [obj.parameters.phantom_name '.m']));
        end
        % ------------------------------------------------------------------
        function create_directory(obj)
            % Create directory to store results.
            
            dir_=split(obj.parameters.path_res, '/');
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
                path=fullfile(obj.parameters.path_res, necessary_dir{k});
                if not(isfolder(path))
                    mkdir(path);
                end 
            end
        end
        % ------------------------------------------------------------------
        function save(obj)
            % Save parameters.
            
            path_to_save=fullfile(obj.parameters.path_res, 'parameters', [obj.parameters.phantom_name '.mat']);
            p=obj.parameters;
            save(path_to_save, 'p');
        end
        % ------------------------------------------------------------------
    end
        
end

