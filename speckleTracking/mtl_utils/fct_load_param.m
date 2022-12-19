function [param]=fct_load_param(path_param)
    
    str = fileread(path_param);
    param = jsondecode(str);
    param.path_res = fct_build_path(split(path_param, '/'), 2);
        
    end
