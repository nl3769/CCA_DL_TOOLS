function fct_display_images(fs, c, depth_phantom, I, dr, num_fig, save, pres)

    dz = c/(2*fs);
    
    ROI_depth = round(depth_phantom / dz);
    ROI = I(1:ROI_depth, :);
    
    if save == false
        figure(num_fig);
    else
        f = figure('visible', 'off');
    end
    
    bmode=abs(hilbert(ROI));
    bmode = 20*log10(bmode/max(bmode(:))) + dr;
    
    imagesc(bmode);
    colormap gray
    colorbar
    caxis([0 dr]) 
    
    if save == true
        pres = fullfile(pres, 'bmode_result', 'bmode.png');
        saveas(f, pres);
    end
    
end