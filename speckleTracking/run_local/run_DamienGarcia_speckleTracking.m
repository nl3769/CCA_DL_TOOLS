close all;
clearvars;

addpath(fullfile('..', 'mtl_cores'))
run(fullfile('..', 'mtl_utils', 'add_path.m'));

%test1()
test2()
%test3()


% -------------------------------------------------------------------------
function test1()

    pdata   = '/home/laine/Desktop/DATA_PREPARATION/tech_001';
    pseq    = fullfile(pdata, 'mat_files', 'images-tech_001.mat');
    pmotion = fullfile(pdata, 'mat_files', 'displacement_field-tech_001.mat');
    pLI     = fullfile(pdata, 'mat_files', 'LI-tech_001.mat');
    pMA     = fullfile(pdata, 'mat_files', 'MA-tech_001.mat');
    pCF     = fullfile(pdata, 'CF-tech_001.txt');
    
    
    
    damienGarcia_speckleTracking(pseq, pmotion, pLI, pMA, pCF)

end

% -------------------------------------------------------------------------
function test2()
    % Fonction circulaure et profil horizontal constant

    height  = 200;
    width   = 200;
    
    T = height; % one period on full-height image
    f = 1/T;
    dz = 3;
    pts = linspace(0, T, T);
    
    col1 = cos(2*pi*f*pts);
    col2 = cos(2*pi*f*(pts + dz));
    
%     apod = hanning(width);
%     apod = repelem(apod, height);
%     apod = reshape(apod', [height, width]);

    % --- real df
    x_df_gt = zeros([height width]);
    z_df_gt = dz * ones([height width]);
%     [x_pos_gt, z_pos_gt] = meshgrid(linspace(1,width, width), linspace(1, height, height));

    % --- create images
    I1 = repelem(col1, width);
    I1 = reshape(I1, [width height])';
    I1 = flipud(I1);
%     I1 = bsxfun(@times, I1, apod);

    % --- motion
    dz = 10;
    dx = 0;
    dX = zeros([height width]) + dx;
    dZ = zeros([height width]) + dz;
    [X_org, Z_org] = meshgrid(linspace(1, width, width), linspace(1, height, height));
    % --- get new image in correct coordinates
    I2 = griddata(X_org + dx, Z_org + dz, I1, X_org, Z_org);

%     I2 = repelem(col2', width);
%     I2 = reshape(I2, [width, height])';
%     I2 = flipud(I2);
    
    figure()
    subplot(2, 2, 1)
    imagesc(I1)
    subplot(2, 2, 2)
    imagesc(I2)
    colormap gray
    subplot(2, 2, 3)
    plot(I1(:,10))
    hold on
    plot(I2(:,10))
    hold off
    legend('I1','I2')
    
        
    % --- compute df
    param           = [];
%     param.winsize   = [32 32; 16 16; 9 9; 5 5];         % size fo the rgion
    param.winsize   = [64 64; 32 32 ; 16 16; 8 8; 5 5];                        % size fo the rgion
    param.iminc     = 1;                                % image increment
    [Dz_pred, Dx_pred, id_z, id_x]   = sptrack(cat(3, I1, I2), param);      % called function from MUST
    
    % --- display result
    display_motion(Dz_pred, Dx_pred, id_z, id_x, 'DG');
    [Dz_gt, Dx_gt, id_z_gt, id_x_gt]    = get_gt(id_x, id_z, cat(3, x_df_gt, z_df_gt));
    display_motion(Dz_gt, Dx_gt, id_z_gt, id_x_gt, 'GT');
    display_motion(Dz_gt-Dz_pred, Dx_gt-Dx_pred, id_z_gt, id_x_gt, 'diff');
    display_superimposed(Dz_pred, Dx_pred, Dz_gt, Dx_gt, id_z, id_x, 'superimpose')
    
    mean_xe = Dx_gt-Dx_pred;
    mean_xe = mean(mean_xe(:));
    mean_ze = Dz_gt-Dz_pred;
    mean_ze = mean(mean_ze(:));

    figure()
    subplot(2,1,1)
    imagesc(Dz_gt-Dz_pred)
    title(['Dz_gt-Dz, mean=' num2str(mean_ze)])
    colorbar
    colormap hot
    subplot(2,1,2)
    imagesc(Dx_gt-Dx_pred)
    title(['Dx_gt-Dx, mean=' num2str(mean_xe)])
    colorbar
    colormap hot

end

% -------------------------------------------------------------------------
function test3()
    % Juste du bruit
    
    height  = 300;
    width   = 500;
    
    I1 = conv2(rand([height width]),ones(5,5),'same');
    [X_org, Z_org] = meshgrid(linspace(1, width, width), linspace(1, height, height));
    
    % --- motion
    dz = 0.1;
    dx = 0.1;
    dX = zeros([height width]) + dx;
    dZ = zeros([height width]) + dz;

    % --- get new image in correct coordinates
    I2 = griddata(X_org + dx, Z_org + dz, I1, X_org, Z_org);
    
    figure()
    subplot(2, 2, 1)
    imagesc(I1)
    subplot(2, 2, 2)
    imagesc(I2)
    colormap gray
    subplot(2, 2, 3)
    plot(I1(:,10))
    hold on
    plot(I2(:,10))
    hold off
    legend('I1','I2')
    
        
    % --- compute df
    param           = [];
%     param.winsize   = [32 32; 16 16; 9 9; 5 5];         % size fo the rgion
    param.winsize   = [64 64; 32 32 ; 16 16; 8 8; 5 5];                        % size fo the rgion
    param.iminc     = 1;                                % image increment
    [Dz_pred, Dx_pred, id_z, id_x]   = sptrack(cat(3, I1, I2), param);      % called function from MUST
    
    % --- display results
    display_motion(Dz_pred, Dx_pred, id_z, id_x, 'DG');
    [Dz_gt, Dx_gt, id_z_gt, id_x_gt]    = get_gt(id_x, id_z, cat(3, dX, dZ));
    display_motion(Dz_gt,           Dx_gt,          id_z_gt,    id_x_gt,    'GT');
    display_motion(Dz_gt-Dz_pred,   Dx_gt-Dx_pred,  id_z_gt,    id_x_gt,    'diff');
    display_superimposed(Dz_pred,   Dx_pred, Dz_gt, Dx_gt,      id_z, id_x, 'superimpose');
    
    mean_xe = Dx_gt-Dx_pred;
    mean_xe = mean(mean_xe(:));
    mean_ze = Dz_gt-Dz_pred;
    mean_ze = mean(mean_ze(:));

    figure()
    subplot(2,1,1)
    imagesc(Dz_gt-Dz_pred)
    title(['Dz_gt-Dz, mean=' num2str(mean_ze)])
    colorbar
    colormap hot
    subplot(2,1,2)
    imagesc(Dx_gt-Dx_pred)
    title(['Dx_gt-Dx, mean=' num2str(mean_xe)])
    colorbar
    colormap hot

end

% -------------------------------------------------------------------------
function [Dz, Dx, id_z, id_x] = get_gt(id_x, id_z, I)
    
    shape = size(id_x);

    % --- indexes computed by the methods
    idx_x = id_x(:);
    idx_z = id_z(:);

    % --- get corresponding GT displacment
    motion_x = I(:, :, 1);
    motion_z = I(:, :, 2);
    
    % --- compute coordinates and extract them from GT
    idx = sub2ind(size(motion_x), idx_z, idx_x);
    Dx = motion_x(idx);
    Dz = motion_z(idx);
    
    % --- reshape
    id_x = reshape(id_x, shape);
    id_z = reshape(id_z, shape);
    Dx = reshape(Dx, shape);
    Dz = reshape(Dz, shape);

end