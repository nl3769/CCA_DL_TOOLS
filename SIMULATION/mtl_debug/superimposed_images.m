clearvars;
close all;
clc; 

% --- DECLARE PARAMETERS
pres_RF_simus = '/home/laine/Desktop/TEST_BIDON/ANS_MAR_SIMUS_2D_SCB/raw_data/RF_raw_data.mat';
pres_PARAM_simus = '/home/laine/Desktop/TEST_BIDON/ANS_MAR_SIMUS_2D_SCB/parameters/parameters.mat';

pres_RF_field = '/home/laine/Desktop/TEST_BIDON/ANS_MAR_FIELD_2D_SCB/raw_data/RF_raw_data.mat';
pres_PARAM_field = '/home/laine/Desktop/TEST_BIDON/ANS_MAR_FIELD_2D_SCB/parameters/parameters.mat';

pphantom = '/home/laine/Desktop/TEST_BIDON/ANS_MAR_FIELD_2D_SCB/phantom/dicom_ANS_MAR_phantom_FIELD_2D_SCB_id_seq_1_id_param_1.mat';

% --- LOAD DATA
RF_field = load_data(pres_RF_field);
[fc_field, fs_field] = load_param(pres_PARAM_field);
RF_simus = load_data(pres_RF_simus);
[fc_simus, fs_simus] = load_param(pres_PARAM_simus);
z_scat = get_scat_pos(pphantom);
c = 1540;

% --- COMPUTE FFT
[fft_field, org_sig_field] = compute_fft(RF_field{64}(:, 96), fc_field, fs_field, c, z_scat, 'field');
[fft_simus, org_sig_simus] = compute_fft(RF_simus{64}(:, 96), fc_simus, fs_simus, c, z_scat, 'simus');

% --- PHANTOM DIFFERENCE
% pres_phantom_simus = '/home/laine/cluster/PROJECTS_IO/SIMULATION/SYNTHETIC_APERTURE/LMO06_image1_RF_ANALYSIS_SIMUS_2022_14_02/phantom/dicom_RF_ANALYSIS_SIMUS_cp_id_seq_1_id_param_1.mat';
% pres_phantom_field = '/home/laine/cluster/PROJECTS_IO/SIMULATION/SYNTHETIC_APERTURE/LMO06_image1_RF_ANALYSIS_FIELD_2022_14_02/phantom/dicom_RF_ANALYSIS_FIELD_cp_id_seq_1_id_param_1.mat';
% 
% phantom_field = load(pres_phantom_field);
% phantom_simus = load(pres_phantom_simus);

% compute_phantom_difference(phantom_field, phantom_simus);

% -------------------------------------------------------------------------
function [RF] = load_data(pres)
    RF = load(pres);
    RF = RF.raw_data;
end
% -------------------------------------------------------------------------
function [fc, fs] = load_param(pres)
    PARAM = load(pres);
    PARAM = PARAM.p;
    
    fc = PARAM.fc;
    fs = fc * PARAM.fsCoef;
end
% -------------------------------------------------------------------------
function [fq_domain, sig] = compute_fft(sig, fc, fs, c, z_scat, name)
       
    % --- DISPLAY
    n = length(sig);

    figure(), clf;
    
    sig = sig / max(sig(:));

    % -- temporal domain
    t = 0:1/fs:(n-1)/fs;
    subplot(2,1,1)
%     plot(t*1e6, sig)  
    plot(t*c/2 * 1e3, sig)
    hold on;
    for i=1:1:length(z_scat)
        line([z_scat(i) * 1e3 z_scat(i) * 1e3], [0 0.5], 'Color', 'red');
    end
    hold off;
    title('Temporal domain')
    xlabel('distance in mm')
    ylabel('RF signal intensity')
    
    % -- frequency domain
    fq_domain = abs(fft(sig));
    PS2= abs(fq_domain/n);
    PS1 = PS2(1:n/2+1);% Single sampling plot
    PS1(2:end-1) = 2*PS1(2:end-1);
    f = fs*(0:(n/2))/n;
    subplot(2,1,2)
    plot(f*1e-6, PS1)
    line([fc*1e-6 fc*1e-6],[0 max(PS1(:))],'Color','red','LineStyle','--')
    title('Fourier domain')
    xlabel('frequency in MHz.')
    ylabel('TF(RF)')
        
    subtitle(['Simulator: ' name])
    
    print(['fourier_' name '.png'], '-dpng');
    
end
% -------------------------------------------------------------------------
function compute_phantom_difference(phantom_field, phantom_simus)

    % --- x
    x_diff = isequal(phantom_field.scatt.x_scatt, phantom_simus.scatt.x_scatt); 
    % --- y
    y_diff = isequal(phantom_field.scatt.y_scatt, phantom_simus.scatt.y_scatt);
    % --- z
    z_diff = isequal(phantom_field.scatt.z_scatt, phantom_simus.scatt.z_scatt);
        
end
% -------------------------------------------------------------------------
function [z_scatt] = get_scat_pos(pphantom)

    phantom = load(pphantom);

    z_scatt = phantom.scatt.z_scatt;
    z_scatt = unique(z_scatt,'rows');
    
end
% -------------------------------------------------------------------------