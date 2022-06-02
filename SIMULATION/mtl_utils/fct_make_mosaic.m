% close all;
% clear all;
% 
% 
% pI1="/home/laine/cluster/Desktop/RESULTS_V1/ANS_MAR_SIMUS_SLB/bmode_result/RF/dicom_ANS_MAR_phantom_SIMUS_SLB_fit_in_vivo.png";
% pI2="/home/laine/cluster/Desktop/RESULTS_V1/ANS_MAR_SIMUS_SLB/bmode_result/RF/dicom_ANS_MAR_phantom_SIMUS_SLB_bmode.png";
% 
% I1 = imread(pI1);
% I2 = imread(pI2);
% 
% dimI1=size(I1);
% dimI2=size(I2);
% if isequal(dimI1, dimI2)
%     mosaic(I1, I2, 100);
% else
%     disp('images are not equal')
% end


function fct_make_mosaic(I1, I2, square_size, pres)
    
    % split image in several patches 
    [height width] = size(I1);
    
    % --- get final dim
    width_f = floor(width / square_size) * square_size;
    height_f = floor(height / square_size) * square_size;   

    demo = get_demo(square_size, width_f, height_f);
    mosaic = get_mosaic(I1, I2, square_size, width_f, height_f);
    

    % --- save results
    
    f = figure('Visible','off');
    imagesc(demo)
    title('Position of simulated/original patches')
    saveas(f, fullfile(pres, ['mosaic_position_size_' num2str(square_size) '.png']));
    close(f)


    f = figure('Visible','off');
    imagesc(mosaic)
    colormap gray
    colorbar
    title('Position of simulated/original patches')
    saveas(f, fullfile(pres, ['mosaic_org_sim_' num2str(square_size) '.png']));
    close(f)


end

function [demo] = get_demo(square_size, width_f, height_f)
    
    pos = [round(square_size/4) round(square_size/3.4)];
    blue_patch = zeros([square_size square_size 3]);
    blue_patch = insertText(blue_patch, pos, 'org', 'BoxColor', 'white');
    red_patch = zeros([square_size square_size 3]);
    red_patch = insertText(red_patch, pos, 'sim', 'BoxColor', 'white');

    demo = zeros([height_f width_f 3]);

    blue_patch(:,:,3) = 100*ones(square_size);

    red_patch(:,:,1) = 100*ones(square_size);
    
    inc=1;

    for i=1:square_size:width_f
        if mod(inc,2) == 0
            b_block = true;
        else
            b_block = false;
        end

        for j=1:square_size:height_f
            if b_block
                demo(j:j+square_size-1, i:i+square_size-1, :) = blue_patch;
                b_block = false;
            else 
                demo(j:j+square_size-1, i:i+square_size-1, :) = red_patch;
                b_block = true;
            end
        end
        inc = inc+1;
    end

end

function [mosaic] = get_mosaic(I1, I2, square_size, width_f, height_f)
   
    inc=1;

    for i=1:square_size:width_f
        if mod(inc,2) == 0
            b_block = true;
        else
            b_block = false;
        end

        for j=1:square_size:height_f
            if b_block
                real_patch = I1(j:j+square_size-1, i:i+square_size-1, :);
                mosaic(j:j+square_size-1, i:i+square_size-1, :) = real_patch;
                b_block = false;
            else 
                sim_patch = I2(j:j+square_size-1, i:i+square_size-1, :);
                mosaic(j:j+square_size-1, i:i+square_size-1, :) = sim_patch;
                b_block = true;
            end
        end
        inc = inc+1;
    end

end