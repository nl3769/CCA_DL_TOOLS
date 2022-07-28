close all;
clearvars;
clc;

set(0, 'DefaultFigureWindowStyle', 'docked');

% --- add path
addpath(fullfile('..', 'mtl_cores/'));
addpath(fullfile('..', 'mtl_utils'));

% --- path to results
path_res='/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER/tech_001';
% path_res='/home/laine/Desktop/NEW_TEST/tech_001/tech_001_id_001_FIELD';
files_ = list_files(path_res);

files=files_;
for id=1:1:length(files)
    pres_ = fullfile(path_res, files{id});
    fct_run_image_reconstruction(pres_, false);
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
