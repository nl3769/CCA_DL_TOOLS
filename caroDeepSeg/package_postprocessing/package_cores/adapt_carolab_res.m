close all;
clearvars;


pseg = "/home/laine/pc/CAROLAB_RESULTS/results/results_ANG_DOM.mat";
res = load(pseg);

% --- get res
seg = res.result.contours_distal;
a=1
%borders = ;