close all;
clear all;

path_res='/home/nlaine/Desktop/debug_simu/RF/';


for k=1:1:128
    
    field=load([path_res 'field-' num2str(k) '.mat']);
    simus=load([path_res 'simus-' num2str(k) '.mat']);

    field=field.apo;
    simus=simus.apo;
    
    figure(1)
    plot(simus)
    hold on
    plot(field)
    legend('simus','field')
    hold off
    title('simus + field')
end

