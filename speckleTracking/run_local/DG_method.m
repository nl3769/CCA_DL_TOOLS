clearvars;
close all;

run(fullfile('..', 'mtl_utils', 'add_path.m'));

pdata = "/home/laine/HDD/PROJECTS_IO/SIMULATION/IMAGENET";
pres = "/home/laine/Desktop/test_DZ_01";

run_DG_method(pdata, pres)