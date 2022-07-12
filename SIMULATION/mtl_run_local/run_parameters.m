close all;
clearvars;

run(fullfile('..', 'mtl_utils', 'add_path.m'));
addpath(fullfile('..', 'mtl_cores'))
% --- generate several sets of parameters according to the desired parameters.
pdatabase = '/home/laine/cluster/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/images';
fnames = load_fnames(pdatabase);

pres = '/home/laine/Desktop/QUASI_RANDOM';
parameters = writeParameters();
info = '';

for i=3:1:3
    acquisition_mode = 'synthetic_aperture '; % scanline_based, synthetic_aperture 
    software = 'SIMUS';

    fct_run_parameters(pdatabase, fnames{i}, pres, info, software, acquisition_mode, '2', 1, 0, 1, 0)

end

% -------------------------------------------------------------------------
 function [fnames] = load_fnames(pres)

     listing = dir(pres);
     incr = 1;

     for id=1:1:length(listing)
             fnames{incr} = listing(id).name;
             incr=incr+1;
     end
 end
 
% -------------------------------------------------------------------------
