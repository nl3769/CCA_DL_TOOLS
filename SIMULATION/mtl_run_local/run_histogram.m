clearvars;
clc;
close all;

load('/home/laine/Documents/PROJECTS_IO/SIMULATION/SYNTHETIC_APERTURE/LMO06_image1_GZ_BF_FIELD_2022_20_01/bmode_result/dicom_GZ_BF_FIELD_cp_id_seq_1_id_param_1/dicom_GZ_BF_FIELD_cp_BF.mat')

G = 0.4;
I = IQ;


% --- TGC
idx = size(I, 1);
tg = ones(idx);
coef = 1/(idx-1);
vec = 1:1:(idx);
time_gain(1:1:(idx), 1) = coef*vec;
TGC = repmat(time_gain, [1, size(I, 2)]);

I = I.* TGC;
pres = '/home/laine/Desktop/tmp'
incr = 1

for dr=30:5:70
    I_DR = 20*log10(I/max(I(:))) + dr;
    % I_DR = uint8(255*I_DR/DR);
    figure(incr)
    imagesc(I_DR)
    axis equal ij tight
    title(['Log compression, DR: ' num2str(dr)])
    caxis([0 dr]) % dynamic range = [-20,0] dB
    colorbar
    colormap gray
    incr = incr+1;

    saveas(gcf, fullfile('/home/laine/Desktop/tmp', strcat('log_compression_', num2str(dr), '.png') ))
end

for gamma=0.1:0.1:1
    I_G = I/max(I(:));
    I_G = I_G.^gamma;
    figure(incr)
    imagesc(I_G)
    axis equal ij tight
    title(['gamma corr.: ' num2str(gamma)])
    colorbar
    colormap gray
    incr = incr+1;
    
    saveas(gcf, fullfile('/home/laine/Desktop/tmp', strcat('gamma_', num2str(gamma), '.png') ))
end

close all;
% I_G_DR = I/max(I(:));
% I_G_DR = I_G_DR.^G;
% I_G_DR = 20*log10(I/max(I_G_DR(:))) + DR;
% figure(3)
% imagesc(I_G_DR)
% axis equal ij tight
% title('gamma corr. + DR')
% colorbar
% colormap gray
