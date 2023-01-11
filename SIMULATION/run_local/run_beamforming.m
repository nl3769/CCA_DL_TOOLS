restoredefaultpath;
close all;
clearvars;
clc;

set(0, 'DefaultFigureWindowStyle', 'docked');
% --- add path
addpath(fullfile('..', 'mtl_cores/'));
addpath(fullfile('..', 'mtl_utils'));
% --- path to results
path_res='/home/laine/Desktop/n03658185_letter_opener';
files = list_files(path_res);
% --- loop over the sequence
for id=1:1:length(files)
    pres_ = fullfile(path_res, files{id});
    fct_run_image_reconstruction(pres_);
end

% -------------------------------------------------------------------------
function [files] = list_files(path)
    % todo -> comment
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
