function [param]=fct_load_param(path_param)
    
    param=load(path_param);
    param=param.p;
    param.path_res = fct_build_path(split(path_param, '/'), 2);
        
    end
