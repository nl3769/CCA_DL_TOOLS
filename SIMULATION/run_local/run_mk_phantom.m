close all;
clearvars;

run(fullfile('..', 'mtl_utils', 'add_path.m'));
addpath(fullfile('..', 'mtl_cores'))

% --- generate several sets of parameters according to the desired parameters.
% pdatabase = '/home/laine/cluster/PROJECTS_IO/DATA/MCMASTER/CROPPED/CAMO_study_cropped/images';
pdatabase = '/home/laine/cluster/PROJECTS_IO/DATA/IMAGENET/sample_00';
pres = '/home/laine/Desktop/MOTION_TEST/V1';
pparam = '/home/laine/Desktop/set_parameters_template.json';
info = '';
fnames = load_fnames(pdatabase);

% -------------------------------------------------------------------------
for i=3:1:5
    fnames{i}
    fct_run_mk_phantom(pdatabase, fnames{i}, pres, pparam, info)
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
