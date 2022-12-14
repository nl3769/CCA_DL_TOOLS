function [X_image, Z_image, X_RF, Z_RF, x_display, z_display, n_pts_x, n_pts_z] = fct_get_grid_2D_elisabeth(phantom, image, probe, dim, dz, param)

    
    % --- bmode image dimension
    z_start = param.remove_top_region * 2; % avoid problem of luminance at the top of the image
    z_end = image.height * image.CF;
    
    x_start = -probe.pitch * (probe.Nelements - 1)/2;
    x_end = probe.pitch * (probe.Nelements - 1)/2;
    
    n_pts_z = round((z_end - z_start) / dz);  
    n_pts_x = probe.Nelements;
    
    % --- compute number of points to beaform data
    x_image = linspace(x_start, x_end, probe.Nelements); 
    z_image = linspace(z_start, z_end, n_pts_z) + param.shift;
    [X_image, Z_image] = meshgrid(x_image, z_image);
    
    x_rf = linspace(x_start, probe.pitch * (probe.Nelements - 1)/2, n_pts_x);
    z_rf = linspace(z_start, dz*dim(1), dim(1)) + param.shift;
    [X_RF, Z_RF] = meshgrid(x_rf, z_rf);
    
    % --- for display
    x_display = [x_start, x_end];
    z_display = [z_start, z_end];
        
end
