function damienGarcia_speckleTracking(varargin)
    
    % --- get parameters
    switch nargin
      case 5
        pseq    = varargin{1};
        pmotion = varargin{2};
        pLI     = varargin{3};
        pMA     = varargin{4};
        pCF     = varargin{5};  
    otherwise
        error('Problem with parameters (fct_run_mk_phantom)')
    end
    
    % --- add path to different functions/libraries
    run(fullfile('..', 'mtl_utils', 'add_path.m'));
    
    % --- get seq
    seq         = load(pseq).data;
    motion_gt   = load(pmotion).data;
    LI          = load(pLI).data;
    MA          = load(pMA).data;

    nb_frame = size(seq, 3);
    
    for id=1:1:nb_frame-1
        I = cat(3, seq(:,:,id), seq(:,:,id+1));
        [Dz, Dx, id_z, id_x]                = compute_motion_DG(I);
        [Dz_gt, Dx_gt, id_z_gt, id_x_gt]    = get_gt(id_x, id_z, motion_gt(:, :, :, id));

        % --- display
        display_motion(Dz, Dx, id_z, id_x, 'DG');
        display_motion(Dz_gt, Dx_gt, id_z_gt, id_x_gt, 'GT');
        display_motion(Dz_gt-Dz, Dx_gt-Dx, id_z_gt, id_x_gt, 'diff');
        display_superimposed(Dz, Dx, Dz_gt, Dx_gt, id_z, id_x, 'superimpose')
% 
        display_axis_motion(motion_gt(:,:,:,id), 1, 'x-motion')
        display_axis_motion(motion_gt(:,:,:,id), 3, 'z-motion')

    end

%     for id=1:1:nb_frame-1
%         I1 = seq(:,:,id);
%         I2 = seq(:,:,id+1);
%         OF = motion_gt(:, :, :, id);
%         
%         coef = 5;
%         I1 = imresize(I1, coef);
%         I2 = imresize(I2, coef);
%         OF = imresize(OF, coef) * coef;
% 
%         I = cat(3, I1, I2);
%         [Dx, Dz, id_x, id_z]                = compute_motion_DG(I);
%         [Dx_gt, Dz_gt, id_x_gt, id_z_gt]    = get_gt(id_x, id_z, OF);
%     
%         % --- display
%         display_motion(Dx, Dz, id_x, id_z, 'DG');
%         display_motion(Dx_gt, Dz_gt, id_x_gt, id_z_gt, 'GT');
%         display_motion(Dx_gt-Dx, Dz_gt-Dz, id_x_gt, id_z_gt, 'diff');
%         display_superimposed(Dx, Dz, Dx_gt, Dz_gt, id_x, id_z, 'superimpose')
%     
%         display_axis_motion(motion_gt(:,:,:,id), 1, 'x-motion')
%         display_axis_motion(motion_gt(:,:,:,id), 3, 'z-motion')
%     end

    compute_motion_DG()
    % --- run example
    example_from_must()
    

end

% -------------------------------------------------------------------------
function [Dz, Dx, id_z, id_x] = compute_motion_DG(pairs)
    
    param           = [];
    param.winsize   = [32 32; 16 16; 9 9; 5 5];                % size fo the rgion
    % param.winsize   = [15 15];                        % size fo the rgion
    param.iminc     = 1;                                % image increment
    [Dz, Dx, id_z, id_x]   = sptrack(pairs,param);      % called function from MUST

end



% -------------------------------------------------------------------------
function [Dz, Dx, id_z, id_x] = get_gt(id_x, id_z, I)
    
    shape = size(id_x);

    % --- indexes computed by the methods
    idx_x = id_x(:);
    idx_z = id_z(:);

    % --- get corresponding GT displacment (
    motion_x = I(:, :, 1);
    motion_z = I(:, :, 3);
    
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

% -------------------------------------------------------------------------
function example_from_must()
    
    % --- EXAMPLE 1
    I1 = conv2(rand(500,500),ones(10,10),'same');
    I2 = imrotate(I1,-3,'bicubic','crop');
    subplot(121), imagesc(I1), axis square off
    title('1^{st} frame')
    subplot(122), imagesc(I2), axis square off
    title('2^{nd} frame')
    colormap gray
    param = [];
    param.winsize = [64 64;32 32];
    [Di,Dj,id,jd] = sptrack(cat(3,I1,I2),param);
    figure
    plot(jd(1:2:end,1:2:end),id(1:2:end,1:2:end),'ko',...
        'MarkerFaceColor','k','MarkerSize',4)
    hold on
    h = quiver(jd(1:2:end,1:2:end),id(1:2:end,1:2:end),...
        Dj(1:2:end,1:2:end),Di(1:2:end,1:2:end),2);
    hold off
    set(h,'LineWidth',1.5)
    axis equal ij tight
    title('This is the motion field')

    % --- EXAMPLE 2
    load('PWI_disk.mat');
    disp('''param'' is a structure whose fields are:')
    disp(param)
    IQ = rf2iq(RF,param);
    dx = 1e-4; % grid x-step (in m)
    dz = 1e-4; % grid z-step (in m)
    [x,z] = meshgrid(-1.25e-2:dx:1.25e-2,1e-2:dz:3.5e-2);
    param.fnumber = []; % an f-number will be determined by DASMTX
    M = dasmtx(1i*size(IQ),x,z,param,'linear');
    spy(M)
    axis square
    title('The sparse DAS matrix')
    tic
    IQb = M*reshape(IQ,[],32);
    IQb = reshape(IQb,[size(x) 32]);
    disp(['Beamforming: time per frame: ' num2str(toc/32*1e3,'%.1f'), ' ms'])
    I = bmode(IQb,30); % B-mode images
    image(x(1,:)*100,z(:,1)*100,I(:,:,1))
    c = colorbar;
    c.YTick = [0 255];
    c.YTickLabel = {'-30 dB','0 dB'};
    colormap gray
    title('The 1^{st} B-mode image')
    ylabel('[cm]')
    axis equal tight ij
    set(gca,'XColor','none','box','off')
    param.ROI = median(I,3)>64;
    param.winsize = [32 32; 24 24; 16 16]; % size of the subwindows
    param.iminc = 1; % image increment
    [Di,Dj,id,jd] = sptrack(I,param);
    image(I(:,:,1))
    colormap gray
    hold on
    h = quiver(jd,id,Dj,Di,3,'r');
    set(h,'LineWidth',1)
    hold off
    title('Motion field (in pix) by speckle tracking')
    axis equal off ij
end