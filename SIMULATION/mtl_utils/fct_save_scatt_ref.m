function fct_save_scatt_ref(scatt, height, pres)

    % --- zoom on few scatterers
    k = 5;
    id = [];

    for w=1:k
        for h=1:k
            id = [id (w-1)*height + h];
        end
    end
    x_scatt_zoom = scatt.x_scatt(id);
    z_scatt_zoom = scatt.z_scatt(id);
    y_scatt_zoom = scatt.y_scatt(id);

    % --- save scatterers map in tiff format
    f=figure('Visible','off');
    set(gcf, 'Position', get(0, 'Screensize'));
    subplot(2,1,1)
    scatter(scatt.x_scatt*1e3, ...
            scatt.z_scatt*1e3, ...
            1, ...
            'filled');
    title('Scatterers reference for flow')

    subplot(2,1,2)
    scatter(x_scatt_zoom*1e3, ...
            z_scatt_zoom*1e3, ...
            15, ...
            'filled');
    title('Zoom')
    pres_ = strcat(pres , '.png');
    saveas(f, pres_);
    close(f)

end
