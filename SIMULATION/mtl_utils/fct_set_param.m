function [path]=fct_set_param(varargin)
    
    switch nargin
        case 2
            experiment=varargin{1};
            param=varargin{2};
        otherwise
            disp('Problem with parameters (createPhantom constructor)')
    end
    

    addpath('../class');

    current_folder=pwd;                                                                 % get current folder
    path_dir=fullfile(current_folder, '..', 'parameters/');                             % where we store parameters
    init_path=strcat(path_dir, 'parameters_templates.m');
    
    % --- we run setParameters (APERTURE)
    switch param
        case "aperture"
            for aperture=8:8:64

                experiment_=strcat(experiment, '_aperture_', num2str(aperture));
                dest_path=strcat(path_dir, experiment_, '.m');

                %%% we modify the .m file in order to have several set of parameters %%%
                % --- we store the content of .m file in a cell
                fid = fopen(init_path, 'r');
                i = 1;
                tline = fgetl(fid);    
                A{i} = tline;
                while ischar(tline)
                    i = i+1;
                    tline = fgetl(fid);
                    A{i} = tline;
                end
                fclose(fid);

                % --- write cell A into .m file by changing the interesting parameters
                fid = fopen(dest_path, 'w');
                for i = 1:numel(A)
                    if A{i+1} == -1
                        fprintf(fid,'%s', A{i});
                        break
                    else
                        if contains(A{i}, 'p.Nactive')
                            A{i}=strcat('p.Nactive', '=', num2str(aperture));
                        elseif contains(A{i}, 'p.phantom_name')
                            A{i}=strcat('p.phantom_name', '=', "'", experiment_, "'");
                        end
                        fprintf(fid,'%s\n', A{i});
                    end
                end

                % --- we run setParamters
                run(dest_path);

                param_obj=setParameters(p, dest_path);

                % --- we delete the file
                delete(dest_path)
            end
            
        case "fnumber"
             for fnumber=0:0.1:2
                if contains(num2str(fnumber), '.')
                    str_=strrep(num2str(fnumber),'.','_') 
                else
                    str_=num2str(fnumber);
                end
                experiment_=strcat(experiment, '_fnumber_', str_);
                disp(['experiment name: ' experiment_]);
                dest_path=strcat(path_dir, experiment_, '.m');

                %%% we modify the .m file in order to have several set of parameters %%%
                % --- we store the content of .m file in a cell
                fid = fopen(init_path, 'r');
                i = 1;
                tline = fgetl(fid);    
                A{i} = tline;
                while ischar(tline)
                    i = i+1;
                    tline = fgetl(fid);
                    A{i} = tline;
                end
                fclose(fid);

                % --- write cell A into .m file by changing the interesting parameters
                fid = fopen(dest_path, 'w');
                for i = 1:numel(A)
                    if A{i+1} == -1
                        fprintf(fid,'%s', A{i});
                        break
                    else
                        if contains(A{i}, 'p.fnumber')
                            A{i}=strcat('p.fnumber', '=', num2str(fnumber), '');
                        elseif contains(A{i}, 'p.phantom_name')
                            A{i}=strcat('p.phantom_name', '=', "'", experiment_, "'");
                        end
                        fprintf(fid,'%s\n', A{i});
                    end
                end

                % --- we run setParamters
                run(dest_path);

                param_obj=setParameters(p, dest_path);

                % --- we delete the file
                delete(dest_path)
             end
        
        case "gamma"
            for gamma=0.1:0.1:1.5
                if contains(num2str(gamma), '.')
                    str_=strrep(num2str(gamma),'.','_') 
                else
                    str_=num2str(gamma);
                end
                experiment_=strcat(experiment, '_gamma_', str_);
                disp(['experiment name: ' experiment_]);
                dest_path=strcat(path_dir, experiment_, '.m');

                %%% we modify the .m file in order to have several set of parameters %%%
                % --- we store the content of .m file in a cell
                fid = fopen(init_path, 'r');
                i = 1;
                tline = fgetl(fid);    
                A{i} = tline;
                while ischar(tline)
                    i = i+1;
                    tline = fgetl(fid);
                    A{i} = tline;
                end
                fclose(fid);

                % --- write cell A into .m file by changing the interesting parameters
                fid = fopen(dest_path, 'w');
                for i = 1:numel(A)
                    if A{i+1} == -1
                        fprintf(fid,'%s', A{i});
                        break
                    else
                        if contains(A{i}, 'p.gamma')
                            A{i}=strcat('p.gamma', '=', num2str(gamma), '');
                        elseif contains(A{i}, 'p.phantom_name')
                            A{i}=strcat('p.phantom_name', '=', "'", experiment_, "'");
                        end
                        fprintf(fid,'%s\n', A{i});
                    end
                end

                % --- we run setParamters
                run(dest_path);

                param_obj=setParameters(p, dest_path);

                % --- we delete the file
                delete(dest_path)
            end
                         
        case 'rangeDB'
             for range_DB=30:5:70
                if contains(num2str(range_DB), '.')
                    str_=strrep(num2str(range_DB),'.','_') 
                else
                    str_=num2str(range_DB);
                end
                experiment_=strcat(experiment, '_rangeDB_', str_);
                disp(['experiment name: ' experiment_]);
                dest_path=strcat(path_dir, experiment_, '.m');

                %%% we modify the .m file in order to have several set of parameters %%%
                % --- we store the content of .m file in a cell
                fid = fopen(init_path, 'r');
                i = 1;
                tline = fgetl(fid);    
                A{i} = tline;
                while ischar(tline)
                    i = i+1;
                    tline = fgetl(fid);
                    A{i} = tline;
                end
                fclose(fid);

                % --- write cell A into .m file by changing the interesting parameters
                fid = fopen(dest_path, 'w');
                for i = 1:numel(A)
                    if A{i+1} == -1
                        fprintf(fid,'%s', A{i});
                        break
                    else
                        if contains(A{i}, 'p.range_DB')
                            A{i}=strcat('p.range_DB', '=', num2str(range_DB), '');
                        elseif contains(A{i}, 'p.phantom_name')
                            A{i}=strcat('p.phantom_name', '=', "'", experiment_, "'");
                        end
                        fprintf(fid,'%s\n', A{i});
                    end
                end

                % --- we run setParamters
                run(dest_path);

                param_obj=setParameters(p, dest_path);

                % --- we delete the file
                delete(dest_path)
            end
                 
        case "copy"

            experiment_=strcat(experiment, '_cp');
            dest_path=strcat(path_dir, experiment_, '.m');

            %%% we modify the .m file in order to have several set of parameters %%%
            % --- we store the content of .m file in a cell
            fid = fopen(init_path, 'r');
            i = 1;
            tline = fgetl(fid);    
            A{i} = tline;
            while ischar(tline)
                i = i+1;
                tline = fgetl(fid);
                A{i} = tline;
            end
            fclose(fid);

            % --- write cell A into .m file by changing the interesting parameters
            fid = fopen(dest_path, 'w');
            for i = 1:numel(A)
                if A{i+1} == -1
                    fprintf(fid,'%s', A{i});
                    break
                else
                    if contains(A{i}, 'p.phantom_name')
                        A{i}=strcat('p.phantom_name', '=', "'", experiment_, "'");
                    end
                    fprintf(fid,'%s\n', A{i});
                end
            end

            % --- we run setParamters
            run(dest_path);

            param_obj=setParameters(p, dest_path);

            % --- we delete the file
            delete(dest_path)
    end
    
        path=param_obj.parameters.path_res;
    end




