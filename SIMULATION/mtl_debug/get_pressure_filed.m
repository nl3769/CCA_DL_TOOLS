close all; clearvars ;

addpath('../MUST_2021/')

probe=getparam('L12-3v');
probe.Nelements=64;
% probe.bandwidth=199;

% focus position (in m)
xf = 0; zf = 2e-2; 

txdel = txdelay(xf,zf,probe); % in s
x = linspace(-4e-2,4e-2,200); % in m
z = linspace(0,6e-2,200); % in m
[x,z] = meshgrid(x,z);
% get pressure field
P = pfield(x,z,txdel,probe);

imagesc(x(1,:)*1e2,z(:,1)*1e2,20*log10(P/max(P,[],'all')))
caxis([-30 0]) % dynamic range = [-20,0] dB
c = colorbar;
c.YTickLabel{end} = '0 dB';
colormap hot
axis equal ij tight
xlabel('x (cm)'), ylabel('z (cm)')
title(['Pressure field ([-30,0] dB) - ' num2str(probe.Nelements) ' elements'])

hold on
plot(xf*1e2,zf*1e2,'bo','MarkerFaceColor','b')
legend('focus point','Location','South')
hold off

