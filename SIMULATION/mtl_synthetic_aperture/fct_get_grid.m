function [X_image, Z_image, Y_image, X_org, Z_org, Y_org, x_display, z_display]=fct_get_grid(param, phantom, image, probe, dim)
    % Returns the corresponding grid to obtain isotropic pixels.            

    % --- bmode image dimension
    z_start=param.shift;                                                        % m
    z_end=phantom.z_max;                                                        % m
    n_points_z=image.height;                                                    % nb of points -> renaùme img

    dim_phantom=phantom.x_max-phantom.x_min;
    dim_probe=probe.pitch*(probe.Nelements-1);
    
    if dim_probe>dim_phantom
        delta=(dim_probe-dim_phantom)/2;
%         x_start=-probe.pitch*(probe.Nelements-1)/2+delta;
%         x_end=probe.pitch*(probe.Nelements-1)/2-delta;
        x_start=phantom.x_min;
        x_end=phantom.x_max;        
        n_points_x=image.width;
    else
        x_start=-probe.pitch*(probe.Nelements-1)/2;
        x_end=-x_start;
        n_points_x=round((x_end-x_start)/image.CF);
    end

    x_image=linspace(x_start, x_end, n_points_x);
    x_org=linspace(-probe.pitch*(probe.Nelements-1)/2, probe.pitch*(probe.Nelements-1)/2, dim(2));

    z_image=linspace(z_start, z_end, n_points_z);
    z_org=linspace(0, dim(1)*probe.c/(2*probe.fs), dim(1));

    [X_image Z_image, Y_image]=meshgrid(x_image, z_image, 1:probe.Nelements);
    [X_org Z_org, Y_org]=meshgrid(x_org, z_org, 1:probe.Nelements); % -> raw_channel_data --> rename

    % --- for display
    x_display=[x_start, x_end];
    z_display=[0, z_end-z_start];

end