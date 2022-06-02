close all;
clearvars;

addpath(fullfile('..', 'function/'));
addpath(fullfile('..', 'class/'));

% --- generate several sets of parameters according to the desired parameters.
pdatabase = '/home/laine/cluster/PROJECTS_IO/DATA/GUILLAUME/original/Sequences/HEALTHY_ANDRE_57';
fnames = load_fnames(pdatabase);

pres = '/home/laine/Desktop/DATA_GAN';
parameters = writeParameters();
info = '';

for i=1:1:1

    acquisition_mode = 'synthetic_aperture '; %scanline_based, synthetic_aperture 
    software = 'FIELD';
    
    % --- generate parameters and phantom
    fct_run_parameters_GAN(pdatabase, fnames{i}, pres, info, software, acquisition_mode, '10')
    
end

% --------------------------------------------------------------------------
 function [fnames] = load_fnames(pres)

     listing = dir(pres);
     incr = 1;

     for id=1:1:length(listing)
         if listing(id).name ~= '.'
             fnames{incr} = listing(id).name;
             incr=incr+1;
         end
     end
 end