close all;
clearvars;
clc;

set(0, 'DefaultFigureWindowStyle', 'docked');

% --- add path
addpath(fullfile('..', 'package_cores/'));

% --- path to results
path_res='/home/laine/cluster/PROJECTS_IO/SIMULATION/CUBS/tech_008';
files_ = list_files(path_res);

% files{1}=files_{3};
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
