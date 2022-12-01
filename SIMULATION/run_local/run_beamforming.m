restoredefaultpath;
close all;
clearvars;
clc;

set(0, 'DefaultFigureWindowStyle', 'docked');
% --- add path
addpath(fullfile('..', 'mtl_cores/'));
addpath(fullfile('..', 'mtl_utils'));

% --- path to results
% path_res = '/home/laine/HDD/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER_V1/tech_001/';
% path_res = '/home/laine/HDD/PROJECTS_IO/SIMULATION/MEIBURGER_1_FRAME/tech_008';
path_res = '/home/laine/HDD/PROJECTS_IO/SIMULATION/CUBS/tech_058';
% path_res='/home/laine/Documents/PROJECTS_IO/SIMULATION/DEBUG/POINT_SCATTERES/sta_field/';
% path_res='/home/laine/Documents/PROJECTS_IO/SIMULATION/DEBUG/STA/POOR_DENSITY/n01496331_electric_ray/';

% path_res='/home/laine/Desktop/NEW_TEST/tech_001/tech_001_id_001_FIELD';
files = list_files(path_res);

for id=1:1:length(files)
    pres_ = fullfile(path_res, files{id});
    fct_run_image_reconstruction(pres_);
    pres_ = fullfile(path_res, files{id})
    fct_run_image_reconstruction(pres_);
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
