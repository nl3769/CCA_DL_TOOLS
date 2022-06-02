close all;
clearvars;


% tmp_=randn(10000000, 1);
tmp_=raylrnd(1,1,10000000)/sqrt(pi/2);
figure
histogram(tmp_)
a=1;