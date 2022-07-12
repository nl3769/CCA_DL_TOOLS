function make_figure_3D_scatt(data_scatt, param, substr)
    % Make figure to display 3D scatteres.
    
    f=figure('visible', 'off');
    set(gcf, 'Position', get(0, 'Screensize'));
    
    x_min = data_scatt.x_min;
    x_max = data_scatt.x_max;
    z_max = data_scatt.z_max;
    z_min = data_scatt.z_min;
    y_max=param.nb_slices * param.slice_spacing * 1e3 * 2;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%% --------------------------
    subplot(2,2,1)
    scatter3([data_scatt.x_scatt; x_min; x_max]*1e3, ...
             [data_scatt.y_scatt; 0; 0]*1e3, ...
             [-data_scatt.z_scatt; -z_min; -z_max]*1e3, ...
             1, ...
             [data_scatt.RC_scatt; 0; 0],...
             'filled')

    xlim([x_min*1e3 x_max*1e3])
    zlim([-z_max*1e3*1.1 0])
    
    ylim([-y_max*2 y_max*2])
    view(-34, 14)
    colormap hot;
    colorbar;
    % Add title and axis labels
    nb_scatt_=length(data_scatt.RC_scatt);
    title(['Phantom (' num2str(nb_scatt_) ' scatterers) -- perspective view'])
    xlabel('x (mm)')
    ylabel('y (mm)')
    zlabel('z (mm)')
    %%%%%%%%%%%%%%%%%%%%%%%% --------------------------
    %%%%%%%%%%%%%%%%%%%%%%%% (x, z)
    subplot(2,2,2)
    scatter3([data_scatt.x_scatt; data_scatt.x_min; data_scatt.x_max]*1e3, ...
             [data_scatt.y_scatt; 0; 0]*1e3, ...
             [-data_scatt.z_scatt; -z_min; -z_max]*1e3, ...
             1, ...
             [data_scatt.RC_scatt; 0; 0],...
             'filled')
    xlim([x_min*1e3 x_max*1e3])
    zlim([-z_max*1e3 0])
    ylim([-y_max*2 y_max*2])
    view(0, 0)
    colormap hot;
    colorbar;
    % Add title and axis labels
    nb_scatt_=length(data_scatt.RC_scatt);
    title(['Phantom (' num2str(nb_scatt_) ' scatterers) -- (x, z) plane'])
    xlabel('x (mm)')
    ylabel('y (mm)')
    zlabel('z (mm)')

    %%%%%%%%%%%%%%%%%%%%%%%% --------------------------
    %%%%%%%%%%%%%%%%%%%%%%%% (x, y)
    subplot(2,2,3)
    scatter3([data_scatt.x_scatt; data_scatt.x_min; data_scatt.x_max]*1e3, ...
             [data_scatt.y_scatt; 0; 0]*1e3, ...
             [-data_scatt.z_scatt; -z_min; -z_max]*1e3, ...
             1, ...
             [data_scatt.RC_scatt; 0; 0],...
             'filled')
    xlim([x_min*1e3 x_max*1e3])
    zlim([-z_max*1e3 0])
    ylim([-y_max*2 y_max*2])
    view(0, 90)
    colormap hot;
    colorbar;
    % Add title and axis labels
    nb_scatt_=length(data_scatt.RC_scatt);
    title(['Phantom (' num2str(nb_scatt_) ' scatterers) -- (x, y) plane'])
    xlabel('x (mm)')
    ylabel('y (mm)')
    zlabel('z (mm)')

    %%%%%%%%%%%%%%%%%%%%%%%% --------------------------
    %%%%%%%%%%%%%%%%%%%%%%%% (z, y)
    subplot(2,2,4)
    scatter3([data_scatt.x_scatt; data_scatt.x_min; data_scatt.x_max]*1e3, ...
             [data_scatt.y_scatt; 0; 0]*1e3, ...
             [-data_scatt.z_scatt; -z_min; -z_max]*1e3, ...
             1, ...
             [data_scatt.RC_scatt; 0; 0],...
             'filled')

    xlim([x_min*1e3 x_max*1e3])
    zlim([-z_max*1e3 0])
    ylim([-y_max*2 y_max*2])
    view(90, 0)
    colormap hot;
    colorbar;
    
    % Add title and axis labels
    nb_scatt_=length(data_scatt.RC_scatt);
    title(['Phantom (' num2str(nb_scatt_) ' scatterers) -- (y, z) plane'])
    xlabel('x (mm)')
    ylabel('y (mm)')
    zlabel('z (mm)')
    
    
    if strcmp(substr, '')
        substr_ = 'original_scatterers_distribution';
    else
        substr_ = param.phantom_name;
    end
    ptmp_ = strcat(substr_, '.png') ;
    saveas(f, fullfile(param.path_res, 'phantom', ptmp_));
    close(f);
end