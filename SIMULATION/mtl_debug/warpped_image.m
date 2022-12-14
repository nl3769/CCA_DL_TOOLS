function warpped_image()
    
    close all;
    clearvars;
    clc;

    pI0     = "/home/laine/Desktop/MotionEstimationDataBaseAnalysisV2/tech_010/tech_010_id_001_FIELD/bmode_result/results/dicom_tech_010_phantom_id_001_FIELD_bmode.png";
    pI1     = "/home/laine/Desktop/MotionEstimationDataBaseAnalysisV2/tech_010/tech_010_id_002_FIELD/bmode_result/results/dicom_tech_010_phantom_id_002_FIELD_bmode.png";
    
    pOF     = "/home/laine/Desktop/MotionEstimationDataBaseAnalysisV2/tech_010/tech_010_id_002_FIELD/phantom/OF_1_2.nii";
    pparam  = "/home/laine/Desktop/MotionEstimationDataBaseAnalysisV2/tech_010/tech_010_id_002_FIELD/phantom/image_information.mat";
    
    x_start = 30;
    x_end   = 30;
    z_start = 40;
    z_end   = 40;

    I0 = imread(pI0);
    I1 = imread(pI1);
    OF = niftiread(pOF);

    param = load(pparam);
    CF = param.image.CF;
    zstart = 2*3e-3;
%     zstart = 0;
    OF = adapt_flow(I0, OF, CF, zstart);

    Iw = warpped(I0, OF);
    
%     I0 = crop(I0, x_start, x_end, z_start, z_end);
%     I1 = crop(I1, x_start, x_end, z_start, z_end);
%     Iw = crop(Iw, x_start, x_end, z_start, z_end);
%     dx = crop(OF(:,:,1), x_start, x_end, z_start, z_end);
%     dz = crop(OF(:,:,3), x_start, x_end, z_start, z_end);

    display(I0, 1, 'I0')
    display(uint8(Iw), 2, 'Iw')
    display(I1, 3, 'I1')
    display_err(double(I1)-Iw, 4, 'diff_warpped')
    display_err(double(I1)-double(I0), 5, 'diff_org')

    
    display_OF(dx, 5, 'dx')
    display_OF(dz, 6, 'dz')

end

% -------------------------------------------------------------------------
function Iw = warpped(I0, OF)

    
    I0 = double(I0);
    [row, col] = size(I0);
    x = linspace(1, col, col);
    z = linspace(1, row, row);
    [X_q, Z_q] = meshgrid(x, z);

    X_w = X_q + OF(:,:,1); 
    Z_w = Z_q + OF(:,:,3);

    Iw = griddata(X_w, Z_w, I0, X_q, Z_q);
%     F = scatteredInterpolant(x,y,v)

end

% -------------------------------------------------------------------------
function display(I, nb, figname)
    
    x0=10;
    y0=10;
    width=550;
    height=400;
    set(gcf,'units','points','position',[x0,y0,width,height])
    I = I(30:end-30, 30:end-30);
    figure(nb)
    imagesc(I)
%     title(figname)
    colormap gray
    cH = colorbar;
    set(cH,'FontSize',30);
    imwrite(I, fullfile("/home/laine/Desktop/warpped", strcat(figname, '.png')))
%     saveas(gcf, fullfile("/home/laine/Desktop", figname), 'png')
%     close()
end

% -------------------------------------------------------------------------
function display_err(I, nb, figname)
    
    x0=10;
    y0=10;
    width=550;
    height=400;
    set(gcf,'units','points','position',[x0,y0,width,height])
     I = I(30:end-30, 30:end-30);
    figure(nb)
    imagesc(I)
%     title(figname)
    colormap hot
    cH = colorbar;
    set(cH,'FontSize',30);

    saveas(gcf, fullfile("/home/laine/Desktop/warpped", figname), 'png')
%     close()
end

% -------------------------------------------------------------------------
function display_OF(I, nb, figname)
    
    x0=10;
    y0=10;
    width=550;
    height=400;
    set(gcf,'units','points','position',[x0,y0,width,height])

    figure(nb)
    imagesc(I)
%     title(figname)
    colormap hot
    cH = colorbar;
    set(cH,'FontSize',30);

    saveas(gcf, fullfile("/home/laine/Desktop/warpped", figname), 'png')
%     close()
end

% -------------------------------------------------------------------------
function I = crop(I, x_start, x_end, z_start, z_end)
    I = I(z_start:end-z_end, x_start:end-x_end);

end

% -------------------------------------------------------------------------
function OF_out = adapt_flow(I, flow, CF, z_start)
    
    [height_I, width_I] = size(I);
    [height_OF_, width_OF_, ~] = size(flow);

    width_roi = linspace(-width_I/2*CF, width_I/2*CF, width_I);
%     height_roi = linspace(0, height_I*CF, height_I) + z_start;
    height_roi = linspace(z_start, height_I*CF+z_start, height_I);
    width_OF = linspace(-width_OF_/2 * CF, width_OF_/2 * CF, width_OF_);
    height_OF = linspace(0, height_OF_ * CF, height_OF_); 
    
    [X,Y]   = meshgrid(width_OF, height_OF);
    [Xq,Yq] = meshgrid(width_roi ,height_roi);
    
    OF_out = zeros(height_I, width_I, 3);
    OF_out(:,:,1) = interp2(X, Y, flow(:,:,1), Xq, Yq);
    OF_out(:,:,2) = interp2(X, Y, flow(:,:,2), Xq, Yq);
    OF_out(:,:,3) = interp2(X, Y, flow(:,:,3), Xq, Yq);
end