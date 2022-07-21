function [phantom]=fct_load_phantom(path_phantom)

    % --- load phantom
    % -> path_phantom: path to load the phantom
    % -> phantom: numeric phantom
    
    
    phantom=load(path_phantom);
    phantom=phantom.scatt;

end