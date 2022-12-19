function fct_mk_figure(Dx, Dz, Dx_gt, Dz_gt, pres, f)
    
    pred = cat(3, Dx, Dz);
    gt = cat(3, Dx_gt, Dz_gt);
    
    pred_norm = vecnorm(pred, 2, 3);
    gt_norm = vecnorm(gt, 2, 3);
    
    % --- compute arg(x, z)
    pred_theta = Dx(:,:,1) ./  pred_norm;
    pred_theta = acos(pred_theta);
    pred_theta = rad2deg(pred_theta);
    
    gt_theta = Dx_gt(:,:,1) ./  gt_norm;
    gt_theta = acos(gt_theta);
    gt_theta = rad2deg(gt_theta);
    

    
    % --- get colorbar magnitude
    % -- x-motion
    colorx = [min(min(Dx(:)), min(Dx_gt(:))), max(max(Dx(:)), max(Dx_gt(:)))];
    % -- z-motion
    colorz = [min(min(Dz(:)), min(Dz_gt(:))), max(max(Dz(:)), max(Dz_gt(:)))];
    % -- norm-motion
    colornorm = [min(min(pred_norm(:)), min(gt_norm(:))), max(max(pred_norm(:)), max(gt_norm(:)))];
    
    subplot(2,4,1);
    imagesc(Dx);
    title('pred Dx');
    colormap(gca,'jet')
    colorbar();
    caxis([colorx(1), colorx(2)]);
    axis off
    
    subplot(2,4,2);
    imagesc(Dz);
    title('pred Dz');
    colormap(gca,'jet')
    colorbar();
    caxis([colorz(1), colorz(2)]);
    axis off
    
    subplot(2,4,3);
    imagesc(pred_norm);
    title(' norm 2 (prediction)');
    colormap(gca,'jet')
    colorbar();
    caxis([colornorm(1), colornorm(2)]);
    axis off
    
    subplot(2,4,4);
    imagesc(gt_theta)
    title("arg(x, z) (gt)")
    caxis([-180 180]); % enforce fixed range for the COLORBAR
    colormap(gca, 'jet');
    colorbar('southoutside');    
    axis off
    axis off
    
    subplot(2,4,5);
    imagesc(Dx_gt);
    title('Dx ref');
    colormap(gca,'jet')
    colorbar();
    caxis([colorx(1), colorx(2)]);
    axis off
    
    subplot(2,4,6);
    imagesc(Dz_gt);
    title('Dz ref');
    colormap(gca,'jet')
    colorbar();
    caxis([colorz(1), colorz(2)]);
    axis off
    
    subplot(2,4,7);
    imagesc(gt_norm);
    title('norm 2 (ref)');
    colorbar();
    colormap(gca,'jet')
    caxis([colornorm(1), colornorm(2)]);
    axis off
    
    subplot(2,4,8);
    imagesc(pred_theta)
    title("arg(x, z) (pred)")
    caxis([-180 180]); % enforce fixed range for the COLORBAR
    colormap(gca, 'jet');
    colorbar('southoutside');    
    axis off

    saveas(f, fullfile(pres, 'motion_disp.png'));
    
end