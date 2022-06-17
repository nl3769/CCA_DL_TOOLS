function [X_image_2D, Z_image_2D, X_org_2D, Z_org_2D, x_display_2D, z_display_2D]=fct_get_grid_2D(param, phantom, image, probe, dim, dz)

    
    % --- bmode image dimension
    z_start     = phantom.z_min;                                                      % m  
    z_end       = phantom.z_max;                                                        % m
    n_points_z  = image.height - round(z_start/dz);                                                    

    dim_phantom = abs(phantom.x_max-phantom.x_min);
    dim_probe   = probe.pitch*(probe.Nelements-1);
    
    if dim_probe > dim_phantom    
        x_start     = phantom.x_min;
        x_end       = phantom.x_max;
        n_points_x  = image.width;
    
    else
        x_start     = -dim_probe;
        x_end       = dim_probe;
        n_points_x  = round((x_end-x_start)/image.CF);
    
    end

    x_image     = linspace(x_start, x_end, n_points_x);
    x_org       = linspace(-dim_probe/2, dim_probe/2, dim(2));

    z_image     = linspace(z_start, z_end, n_points_z);
    z_org       = linspace(dz, dim(1) * dz, dim(1));
    
    [X_image_2D, Z_image_2D] = meshgrid(x_image, z_image);
    [X_org_2D, Z_org_2D]     = meshgrid(x_org, z_org); 

    % --- for display
    x_display_2D = [x_start, x_end];
    z_display_2D = [z_start, z_end];

end
