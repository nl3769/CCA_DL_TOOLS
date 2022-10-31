function fct_save_flow(flow, pres)
    
    % --- sqrt(x^2 + z^2)
    xz_2 = sqrt(flow(:,:,1).^2 + flow(:,:,3).^2);
    % --- sqrt(x^2 + z^2 + y^2)
    xyz_2 = sqrt(flow(:,:,1).^2 + flow(:,:,2).^2 + flow(:,:,3).^2);
    
    % --- compute arg(x, z)
    theta = flow(:,:,1) ./  xz_2;
    theta = acos(theta);
    theta = rad2deg(theta);

    dim = size(theta);
    clb_int = max(max(xz_2(:)), max(xyz_2(:)));
    % --- add negative angles
    for i=1:1:dim(2)
        for j=1:1:dim(1)
            if theta(j,i) > 180
                theta(j,i) = 180 - theta(j,i);
            end
        end
    end
    
    f=figure('visible', 'off');
    set(gcf, 'Position', get(0, 'Screensize'));
    % ------
    % ------
    subplot(3,2,1)
    imagesc(xz_2)
    title("$ \sqrt((x2 - x1)^2 + (y2 - y1)^2) $",'interpreter','latex')
    caxis([-clb_int, clb_int])
    colormap(gca,'jet')
    colorbar
    axis off
    % ------
    % ------
    subplot(3,2,3)
    imagesc(xyz_2)
    title("$ \sqrt((x2 - x1)^2 + (y2 - y1)^2) + (z2 - z1)^2) $",'interpreter','latex')
    caxis([-clb_int, clb_int])
    colormap(gca,'jet')
    colorbar
    axis off
    % ------
    % ------
    subplot(3,2,[2, 4])
    imagesc(theta)
    title("$ arg(x, y) $",'interpreter','latex', 'FontSize', 30) 
    colormap(gca, 'jet');
    caxis([-180 180]); % enforce fixed range for the COLORBAR
    colorbar('southoutside');    
    axis off
    % ------
    % ------
    subplot(3,2,5)
    imagesc(flow(:,:,1))
    title("$ x_{displacement} $",'interpreter','latex', 'FontSize', 30) 
    colormap(gca, 'jet');
    colorbar('southoutside');    
    axis off
    % ------
    % ------
    subplot(3,2,6)
    imagesc(flow(:,:,3))
    title("$ z_{displacement} $",'interpreter','latex', 'FontSize', 30) 
    colormap(gca, 'jet');
    colorbar('southoutside');    
    axis off
    
    saveas(f, strcat(pres, '.png'));
    close(f);

    % niftiwrite(V,'outbrain.nii');


    %     colormap bluewhitered
end
