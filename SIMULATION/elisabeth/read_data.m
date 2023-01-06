close all; clearvars;
run(fullfile('..', 'mtl_utils', 'add_path.m'))

% --- only one sample
path = '/home/laine/Desktop/ELISABETH/n01748264_Indian_cobra';

pI0 = fullfile(path, 'n01748264_Indian_cobra_id_001_FIELD_3D/bmode_result/results/dicom_n01748264_Indian_cobra_phantom_id_001_FIELD_3D_bmode.png');
pI1 = fullfile(path, 'n01748264_Indian_cobra_id_002_FIELD_3D/bmode_result/results/dicom_n01748264_Indian_cobra_phantom_id_002_FIELD_3D_bmode.png');
pOF = fullfile(path, 'n01748264_Indian_cobra_id_002_FIELD_3D/phantom/OF_1_2.nii');
pinfo = fullfile(path, 'n01748264_Indian_cobra_id_002_FIELD_3D/phantom//image_information.mat');
pRF0 = fullfile(path, 'n01748264_Indian_cobra_id_001_FIELD_3D/raw_data');
pRF1 = fullfile(path, 'n01748264_Indian_cobra_id_002_FIELD_3D/raw_data');
pparam = fullfile(path, 'n01748264_Indian_cobra_id_001_FIELD_3D/parameters/parameters.json');
pprobe = fullfile(path, 'n01748264_Indian_cobra_id_001_FIELD_3D/raw_data/dicom_n01748264_Indian_cobra_phantom_id_001_FIELD_3D_probe.mat');

% --- read data (images + motion)
I0 = imread(pI0);
I1 = imread(pI1);
OF = niftiread(pOF);
param=fct_load_param(pparam);
probe = load(pprobe);
probe=probe.probe;

% --- adapt data (motion to fit image dimension)
iinfo = load(pinfo);
CF = iinfo.image.CF;
zstart = 2*param.remove_top_region;
OF = adapt_flow(I0, OF, CF, zstart);

% --- load RF
RF0 = fct_run_cluster_RF(pRF0);
RF0 = fct_rf_padding(RF0, probe);
RF1 = fct_run_cluster_RF(pRF1);
RF1 = fct_rf_padding(RF1, probe);


Iw = warpped(I0, OF);
display(I0, 1, 'I0')
display(uint8(Iw), 2, 'Iw')
display(I1, 3, 'I1')
display_err(double(I1)-Iw, 4, 'diff warpped')
display_err(double(I1)-double(I0), 5, 'diff org')


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
    title(figname)
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
    title(figname)
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
function [RF_out] = fct_rf_padding(RF, probe)
    
%     RF_padding = RF;
    dt = 1/(2*probe.fs);
    
    for id_apert=1:size(RF, 2)
    	toffset = RF{id_apert}(1,1);
        nb_line_padding = round(toffset/dt);
        padding = zeros(nb_line_padding, 192);
        RF_padding{id_apert} = [padding; RF{id_apert}(2:end,:)];
    end
    
    % --- get height max
    height = 0;
    for id_apert=1:size(RF_padding, 2)
    	if height < size(RF_padding{id_apert}, 1) 
        	height=size(RF_padding{id_apert}, 1); 
        end
    end
    
    % --- get number of emit transducer
    Nelements = size(RF{1}, 2);
    
	for id_apert=1:size(RF, 2)
    	nb_line_padding=height-size(RF_padding{id_apert}, 1);
        padding=zeros(nb_line_padding, Nelements);
        RF_padding_f{id_apert}=[RF_padding{id_apert}(2:end,:) ; padding];
    end
    
    RF_out = zeros(size(RF_padding_f{1}, 1), size(RF_padding_f{1}, 2), Nelements);
	for id_apert=1:size(RF, 2)
        RF_out(:,:,id_apert) = RF_padding_f{id_apert};
    end
end