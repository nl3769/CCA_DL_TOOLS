clearvars;
close all;

run(fullfile('..', 'mtl_utils', 'add_path.m'));

pdata = "/home/laine/HDD/PROJECTS_IO/SIMULATION/IMAGENET";
pres = "/home/laine/Documents/PROJECTS_IO/MOTION/BASELINE/DG_METHOD";

run_DG_method(pdata, pres)