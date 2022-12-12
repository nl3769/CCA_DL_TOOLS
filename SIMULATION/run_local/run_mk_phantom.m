close all;
clearvars;

run(fullfile('..', 'mtl_utils', 'add_path.m'));
addpath(fullfile('..', 'mtl_cores'))

% --- generate several sets of parameters according to the desired parameters.
pdatabase = '/home/laine/cluster/PROJECTS_IO/DATA/SIMULATION/MEIBURGER/images';
% pdatabase = '/home/laine/Documents/PROJECTS_IO/DATA/PERSONAL_IMAGES';
pres = '/home/laine/MK_PHANTOM_TEST';
pparam = '/home/laine/Documents/REPOSITORIES/CCA_DL_TOOLS/SIMULATION/parameters/set_parameters_template.json';
info = '';
fnames = load_fnames(pdatabase);

% -------------------------------------------------------------------------
for i=10:1:12
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