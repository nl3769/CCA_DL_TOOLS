close all;
clearvars;

path_img='/home/laine/Documents/PROJECTS_IO/DATA/SEGMENTATION/GUILLAUME/Sequences/HEALTHY_ANDRE_57/ANG_DOM';
%     probe_frequency=info.SequenceOfUltrasoundRegions.Item_1.TransducerFrequency;
info=dicominfo(path_img);
rect = [info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0+10,...
        info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0+50,...
        info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1-100,...
        info.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1-120];

% load DICOM image
image=dicomread(path_img, 'frame', 1); % we load the first frame of the sequence
image=rgb2gray(imcrop(image,rect));
image=imresize(image, 0.5);
addpath('../function/')

y_cut = 40;

f=figure('visible', 'off');
imagesc(image);
hold on
line([150, 150],[1, size(image, 1)]);
hold off
colormap gray;
title('original image');
saveas(f, 'results_filtering/original_image.png');

f=figure('visible', 'off');
plot(image(150, :))
title('profil original image');
saveas(f, 'results_filtering/profil_original_image.png');


for iter=5:1:10
    sigma_s=.4;
% for neigh=3:2:15
    neigh=3;
    filtered_image=fct_bilateral_filtering(image, 10, sigma_s, neigh, iter);
    f=figure('visible', 'off');
    imagesc(filtered_image)
    hold on
    line([150, 150],[1, size(image, 1)]);
    hold off
    colormap gray;
    title(['sigma_s: ' num2str(sigma_s)]);
    saveas(f, ['results_filtering/sigma_s_' num2str(sigma_s) '_iter_' num2str(iter) '_neigh_' num2str(neigh) '.png']);
    
    f=figure('visible', 'off');
    plot(filtered_image(150, :))
    title('profil');
    saveas(f, ['results_filtering/profil_sigma_s_' num2str(sigma_s) '_iter_' num2str(iter) '_neigh_' num2str(neigh) '.png']);
end

% --- voisinage
% aller au dela de 3 ne change rien
