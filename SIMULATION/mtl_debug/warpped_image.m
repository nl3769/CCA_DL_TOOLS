function warpped_image()
    
    close all;
    clearvars;
    clc;

    pI0 = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/SYNTHETIC_DATASET/IMAGENET/n02229544_cricket/dicom_n02229544_cricket_phantom_id_057.png";
    pI1 = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/SYNTHETIC_DATASET/IMAGENET/n02229544_cricket/dicom_n02229544_cricket_phantom_id_058.png";
    pOF = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/SYNTHETIC_DATASET/IMAGENET/n02229544_cricket/OF_57_58.nii";
    pCF = "/home/laine/cluster/PROJECTS_IO/DATA/OPTICAL_FLOW/SYNTHETIC_DATASET/IMAGENET/n02229544_cricket/CF.txt";
    
    x_start = 30;
    x_end = 30;
    z_start = 40;
    z_end = 40;

  

    I0 = imread(pI0);
    I1 = imread(pI1);
    OF = niftiread(pOF);
    CF = readCF(pCF);   
    Iw = warpped(I0, OF);
    
    
    I0 = crop(I0, x_start, x_end, z_start, z_end);
    I1 = crop(I1, x_start, x_end, z_start, z_end);
    Iw = crop(Iw, x_start, x_end, z_start, z_end);
    dx = crop(OF(:,:,1), x_start, x_end, z_start, z_end);
    dz = crop(OF(:,:,3), x_start, x_end, z_start, z_end);

    display(I0, 1, 'I0')
    display(Iw, 2, 'Iw')
    display(I1, 3, 'I1')
    display_err(double(I1)-Iw, 4, 'diff')

    
    display_OF(dx, 5, 'dx')
    display_OF(dz, 5, 'dz')

end

% -------------------------------------------------------------------------
function [CF] = readCF(pCF)

    fileID = fopen(pCF,'r');
    CF = fscanf(fileID, '%f');
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

    figure(nb)
    imagesc(I)
%     title(figname)
    colormap gray
    cH = colorbar;
    set(cH,'FontSize',30);
    saveas(gcf, fullfile("/home/laine/Desktop", figname), 'png')
    close()
end

% -------------------------------------------------------------------------
function display_err(I, nb, figname)
    
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

    saveas(gcf, fullfile("/home/laine/Desktop", figname), 'png')
    close()
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

    saveas(gcf, fullfile("/home/laine/Desktop", figname), 'png')
    close()
end

% -------------------------------------------------------------------------
function I = crop(I, x_start, x_end, z_start, z_end)
    I = I(z_start:end-z_end, x_start:end-x_end);

end
