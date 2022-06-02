close all;
clear all;

path_res='/home/nlaine/Desktop/debug_simu/delay/';


for k=1:1:128
    
    field=load([path_res 'field-' num2str(k) '.mat']);
    simus=load([path_res 'simus-' num2str(k) '.mat']);

    field=field.delay;
    simus=simus.delay;
    
    figure(1)
    plot(field)
    hold on
    plot(simus-max(simus))
    hold off
    title('simus + field')
    
end

