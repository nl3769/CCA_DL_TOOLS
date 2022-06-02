function fct_analysis(psave, pres_org, pres_sim, x_disp, z_disp)
    close all;
    % --- load img
    I_org = load_image(pres_org);
    I_sim = load_image(pres_sim);
    
    % --- adjust histogram
    I_org = fct_expand_histogram(I_org, 0, 255);
    I_sim = fct_expand_histogram(I_sim, 0, 255);

    % --- get dim
    [height_org, width_org] = size(I_org);
    [height_sim, width_sim] = size(I_sim);
    
    x_axis = linspace(x_disp(1), x_disp(2), width_org) ;
    z_axis = linspace(z_disp(1), z_disp(2), height_org);

    % --- profil
    [x_prof_org, y_prof_org] = plot_profil(psave, round(height_org/2), round(width_org/2), I_org, 'original', x_axis, z_axis);
    [x_prof_sim, y_prof_sim] = plot_profil(psave, round(height_sim/2), round(width_sim/2), I_sim, 'simulation', x_axis, z_axis);
    
    % --- cluster profil
    cluster_profil(x_prof_org, y_prof_org, x_prof_sim, y_prof_sim, psave, x_axis, z_axis);

    % --- plot histogram
    z_s = max(z_disp) * 4/5;
    dz =  2 / 1000; 
    
    x_s = 0;
    dx =  2 / 1000; 
    
    plot_histogram(I_org, x_s, dx, z_s, dz, psave, 'original', x_axis, z_axis);
    plot_histogram(I_sim, x_s, dx, z_s, dz, psave, 'simulation', x_axis, z_axis);
    
    % --- mosaic
    fct_make_mosaic(I_org, I_sim, 10, psave);
    fct_make_mosaic(I_org, I_sim, 25, psave);
    fct_make_mosaic(I_org, I_sim, 50, psave);
    fct_make_mosaic(I_org, I_sim, 100, psave);
    fct_make_mosaic(I_org, I_sim, 150, psave);

    % --- image division
    divid_images(I_org, I_sim, psave, x_axis, z_axis)

    % --- image difference
    difference_images(I_org, I_sim, psave, x_axis, z_axis)
end

% -------------------------------------------------------------------------
function [I] = load_image(pres)
    
    I = imread(pres);
    
    if length(I) == 3
        rgb2gray(I);
    end

end

% -------------------------------------------------------------------------
function [x_prof, y_prof] = plot_profil(pres, y_p, x_p, I, name, x_axis, z_axis)
        
    [height, width] = size(I);
    
    x_prof = I(y_p, :);
    y_prof = I(:, x_p);

    f = figure('visible', 'off');
    % ---
    subplot(2,2,[1,2])
    imagesc(x_axis*1e3, z_axis*1e3, I);
    colormap gray
    colorbar
    hold on;
    plot(zeros(length(z_axis), 1), z_axis*1e3, 'Color','red', 'linewidth', 1);
    plot(x_axis *1e3, mean(z_axis) * 1e3*ones(length(x_axis), 1), 'Color','green', 'linewidth', 1);
%     line([x_p, x_p], [1 height], 'Color','red', 'linewidth', 1);
%     line([1, width], [y_p y_p],'Color','green', 'linewidth', 1);
    
    hold off;
    title(name)
    xlabel('width in mm')
    ylabel('height in mm')
    % ---
    subplot(2,2,3)
    plot(z_axis*1e3, y_prof, 'Color','red');
    title('Vertical profil')
    xlabel('height in mm')
    ylabel('gray level (0-255)')
    % ---
    subplot(2,2,4)
    plot(x_axis*1e3, x_prof, 'Color','green');
    title('horizontal profil')
    xlabel('width in mm')
    ylabel('gray level (0-255)')

    % --- save figure

    saveas(f, fullfile(pres, ['profile_' name '.png']));
    close(f)
end

% -------------------------------------------------------------------------
function cluster_profil(x_prof_org, y_prof_org, x_prof_sim, y_prof_sim, pres, x_axis, z_axis)
    
    % ---- HORIZONTAL PROFIL
    f = figure('visible', 'off');
    % ---
    subplot(3,1,1)
    plot(x_axis*1e3, x_prof_org, 'Color','red', 'linewidth', 1)
    title('original')
    % ---
    subplot(3,1,2)
    plot(x_axis*1e3, x_prof_sim, 'Color','green', 'linewidth', 1)
    title('simulation')
    % ---
    subplot(3,1,3)
    plot(x_axis*1e3, x_prof_org, 'Color','red', 'linewidth', 1)
    hold on;
    plot(x_axis*1e3, x_prof_sim, 'Color','green', 'linewidth', 1)
    hold off;
    legend('original', 'simulation')
    xlabel('width in mm')
    ylabel('gray level (0-255)')
    sgtitle('horizontal profil')

    saveas(f, fullfile(pres, 'horizontal_profil.png'))
    close(f)

        % ---- VERTICAL PROFIL
    f = figure('visible', 'off');
    % ---
    subplot(3,1,1)
    plot(z_axis*1e3, y_prof_org, 'Color','red', 'linewidth', 1)
    title('original')
    % ---
    subplot(3,1,2)
    plot(z_axis*1e3, y_prof_sim, 'Color','green', 'linewidth', 1)
    title('simulation')
    % ---
    subplot(3,1,3)
    plot(z_axis*1e3, y_prof_org, 'Color','red', 'linewidth', 1)
    hold on;
    plot(z_axis*1e3, y_prof_sim, 'Color','green', 'linewidth', 1)
    hold off;
    legend('original', 'simulation')
    title('Simulation + orignal ')
    xlabel('height in mm')
    ylabel('gray level (0-255)')
    sgtitle('vertical profil')

    saveas(f, fullfile(pres, 'vertical_profil.png'))
    close(f)
end

% ------------------------------------------------------------------------
function plot_histogram(I, x_s, dx, z_s, dz, pres, name, x_axis, z_axis)
    
    % --- find x_s/x_e/z_s/z_e in pixel coordinates
    dim_x = (x_axis(end) - x_axis(1)) / length(x_axis);
    dim_z = (z_axis(end) - z_axis(1)) / length(z_axis);
    
    pix_x_s = round((x_s + max(x_axis))/dim_x);
    pix_x_e = round(((x_s + dx) + max(x_axis))/dim_x);
    
    pix_z_s = round((z_s)/dim_z);
    pix_z_e = round((z_s + dz)/dim_z);

    x_box = linspace(x_s, x_s + dx, pix_x_e-pix_x_s);
    z_box = linspace(z_s, z_s + dz, pix_z_e-pix_z_s);

    I_crop = I(pix_x_s:pix_x_e, pix_z_s: pix_z_e);
    f = figure('visible', 'off');
    % ---
    subplot(2,2,1)
    imagesc(x_axis * 1e3, z_axis * 1e3, I)
    colormap gray
    colorbar
    hold on;
    
    plot(x_box * 1e3, z_s*ones(length(x_box), 1)  * 1e3, 'Color','red', 'linewidth', 1);
    plot(x_box * 1e3, (z_s+dz)*ones(length(x_box), 1) * 1e3, 'Color','red', 'linewidth', 1);

    plot(x_s * ones(length(z_box), 1) * 1e3, z_box * 1e3, 'Color','red', 'linewidth', 1);
    plot((x_s + dx) * ones(length(z_box), 1) * 1e3, z_box * 1e3, 'Color','red', 'linewidth', 1);
    
%     line([x_e x_s], [y_e y_e], 'Color','red', 'linewidth', 1)
%     line([x_e x_s], [y_s y_s], 'Color','red', 'linewidth', 1)
%     line([x_e x_e], [y_e y_s], 'Color','red', 'linewidth', 1)
%     line([x_s x_s], [y_e y_s], 'Color','red', 'linewidth', 1)
    hold off;
    xlabel('width in mm')
    ylabel('height in mm')
    
    % ---
    subplot(2,2,3)
    imagesc(x_box*1e3, z_box*1e3, I_crop)
    colormap gray
    colorbar
    xlabel('width in mm')
    ylabel('height in mm')
    % ---
    subplot(2,2,[2,4])
    histogram(I_crop)
    xlabel('gray level')
    ylabel('number of pixels')
    
    % --- save result
    saveas(f, fullfile(pres, ['histogram_' name '.png']));
    close(f)
end

% -------------------------------------------------------------------------
function divid_images(Iorg, Isim, pres, x_axis, z_axis)
	
	div = double(Iorg) ./ (double(Isim) + eps);
    div( div > 10) = 10;
	
	f = figure('visible', 'off');
	imagesc(x_axis * 1e3, z_axis * 1e3, div);
	title('$D = \frac{I_{org}}{I_{sim}}, D(D>10)=10$','Interpreter','latex')
	xlabel('Width in mm')
	ylabel('Height in mm')
	colormap hot;
	colorbar;
	saveas(f, fullfile(pres, 'Iorg_over_Isim.png'));
	close(f)
    
end
% -------------------------------------------------------------------------
function difference_images(Iorg, Isim, pres, x_axis, z_axis)
	
	diff = Iorg - Isim;
	
	f = figure('visible', 'off');
	imagesc(x_axis * 1e3, z_axis * 1e3, diff);
	title('$|I_{org} - I_{sim}|$','Interpreter','latex')
	xlabel('Width in mm')
	ylabel('Height in mm')
	colormap hot;
	colorbar;
	saveas(f, fullfile(pres, 'Iorg_minus_Isim.png'));
	close(f)
    
end

% -------------------------------------------------------------------------