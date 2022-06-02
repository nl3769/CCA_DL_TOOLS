function [X_image_2D, Z_image_2D, X_org_2D, Z_org_2D, x_display_2D, z_display_2D]=fct_get_grid_2D(param, phantom, image, probe, dim)

    
    % --- bmode image dimension
%     z_start=phantom.z_min;                                                      % m
    z_start=1e-3;                                                      % m
%     z_satrt = param.remove_top_region;  
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
%     z_org=linspace(param.remove_top_region, dim(1)*probe.c/(2*probe.fs), dim(1));
    z_org=linspace(0, dim(1)*probe.c/(2*probe.fs), dim(1));
    [X_image_2D Z_image_2D]=meshgrid(x_image, z_image);
    [X_org_2D Z_org_2D]=meshgrid(x_org, z_org); % -> raw_channel_data --> rename

    % --- for display
    x_display_2D=[x_start, x_end];
    z_display_2D=[z_start, z_end];

end
