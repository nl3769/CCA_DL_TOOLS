close all;
clear all;

path_res='/home/laine/Documents/PROJECTS_IO/SIMULATION/SYNTHETIC_APERTURE/ANG_DOM_SYNTHETIC_APERTURE_12_15_2021/raw_data/';


field=load([path_res 'dicom_ANG_DOM_synthetic_aperture_cp_1_raw_data_field_single_scatt.mat']);
simus=load([path_res 'dicom_ANG_DOM_synthetic_aperture_cp_1_raw_data.mat']);
field=field.raw_data;
simus=simus.raw_data;

field_sig=field{1};
simus_sig=simus{1};

lim=min(size(field_sig, 1), size(simus_sig, 1));
field_sig=field_sig(1:lim,:);
simus_sig=simus_sig(1:lim,:);
plot_rf(field_sig, 'field')
plot_rf(simus_sig, 'simus')
% plot_rf(simus{4})

function[]=plot_rf(RF, name)
    N_elements=64;
    fc=7500000;
    fs=fc*6;
    t = (0:size(RF,1)-1)/fs*1e6; % in microseconds
    figure
    subplot(211)
    [N,M]=size(RF);
    RF=RF/max(max(RF))*4;
    
    for i=1:8:N_elements
        plot((0:N-1)/fs+t,RF(:,i)+i), hold on
    end
    
    hold off
    title(['Individual traces [' name ']'])
    xlabel('Time [us]')
    ylabel('Normalized response')
    subplot(212)
    plot((0:N-1)/fs+t,sum(RF'))
    title('Summed response')
    xlabel('Time [us]')
    ylabel('Normalized response')
end

