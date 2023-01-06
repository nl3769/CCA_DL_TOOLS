close all;
clearvars;
clc;

set(0, 'DefaultFigureWindowStyle', 'docked');

% --- add path
addpath(fullfile('..', 'mtl_cores/'));
addpath(fullfile('..', 'mtl_utils'));

% --- path to results
path_res01='/home/laine/cluster/PROJECTS_IO/SIMULATION/GAN_DATA';
patients = list_files(path_res01);

for idp=1:1:length(patients)
%parfor idp=21:1:length(patients)
    files = list_files(fullfile(path_res01, patients{idp}))
    
    for id=1:1:length(files)
        pres_ = fullfile(path_res01, patients{idp}, files{id});
        fct_run_image_reconstruction(pres_);
    end
end
    
% -------------------------------------------------------------------------
function [files] = list_files(path)

    listing = dir(path);
    incr=1;
    
    for id=1:1:length(listing)
        if listing(id).name ~= '.'
            files{incr} = listing(id).name;
            incr=incr+1;
        end
    end

end
% -------------------------------------------------------------------------