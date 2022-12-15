function [X_image, Z_image, X_RF, Z_RF, x_display, z_display, n_pts_x, n_pts_z]=fct_get_grid_2D(phantom, image, probe, dim, dz, param)

    
    % --- bmode image dimension
    z_start = param.remove_top_region * 2; % avoid problem of luminance at the top of the image
    z_end = image.height * image.CF;
    n_pts_z = image.height - ceil(z_start/image.CF);                                                    

    dim_phantom = phantom.x_max-phantom.x_min;
    dim_subprobe = probe.pitch * (probe.Nelements - param.Nactive); 
%     dim_subprobe = probe.pitch * (probe.Nelements - 1); 
    if dim_subprobe > dim_phantom    
        x_start     = -image.width * image.CF / 2;
        x_end       = image.width * image.CF / 2;
        n_pts_x  = image.width;
    
    else
        x_start     = -dim_subprobe/2;
        x_end       = dim_subprobe/2;
        n_pts_x  = round((x_end-x_start)/image.CF);
    end
    
    
    % --- make grid
    x_image = linspace(x_start, x_end, n_pts_x);
    z_image = linspace(z_start, z_end, n_pts_z) + param.shift;
    [X_image, Z_image] = meshgrid(x_image, z_image);
    
    x_rf = linspace(-probe.pitch * (probe.Nelements - 1)/2, probe.pitch * (probe.Nelements - 1)/2, probe.Nelements);
    z_rf = linspace(z_start, dz*dim(1), dim(1)) + param.shift;
    [X_RF, Z_RF] = meshgrid(x_rf, z_rf);
    
    % --- for display
    x_display = [x_start, x_end];
    z_display = [z_start, z_end];

end
