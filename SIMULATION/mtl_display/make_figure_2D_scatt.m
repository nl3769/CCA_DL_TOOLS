function make_figure_2D_scatt(data_scatt, param, substr)
    % Make figure to display 3D scatteres.

    % --- save scatterers map in png format
    idx=find(data_scatt.y_scatt==0);
    f=figure('visible', 'off');
    set(gcf, 'Position', get(0, 'Screensize'));
    scatter(data_scatt.x_scatt(idx) * 1e3, ...
            -data_scatt.z_scatt(idx) * 1e3, ...
            30, ...
            data_scatt.RC_scatt(idx), ...
            'filled');

    xlim([data_scatt.x_min*1e3 data_scatt.x_max*1e3])
    ylim([-data_scatt.z_max*1e3 -data_scatt.z_min*1e3])
    colormap hot;
    colorbar;

    nb_scatt_=size(data_scatt.x_scatt, 1);
    
    
    
    title(['Phantom (' num2str(nb_scatt_) ' scatterers) - ' substr]), xlabel('Azimuth [mm]'), ylabel('Depth [mm]',  'Interpreter', 'none');
    
    if strcmp(substr, '')
        phname = 'original';
    else
        phname = param.phantom_name;
    end
%     ptmp = strcat(phname, substr, '.svg');
    ptmp = strcat(phname, substr, '.png');
    saveas(f, fullfile(param.path_res, 'phantom', ptmp));
    close(f);

end