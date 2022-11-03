close all;
clearvars;


% gaussian parameters
A = 1;
mu_x = -10;
mu_z = 0;
sigma_x = 20;
sigma_z = 40;
min = -25;
max = 25;
nb_pts = 1000;

% time parameters
T = 4*pi;
dt = T/50;
t = 0:dt:T;

% grid
x = linspace(min,max,nb_pts);
z = linspace(min,max,nb_pts);
[X, Z] = meshgrid(x, z);

gauss = A * exp(-((X - mu_x).^2 / (2*sigma_x^2) + (Z - mu_z).^2 / (2*sigma_z^2)));
[Gx,Gy] = imgradientxy(gauss);

A0 = 1;
A1 = 1;
A2 = 1;
A3 = 1;

for i=1:1:size(t,2)
    
    mu_x_0 = -15 + 1*sin(t(i));
    mu_z_0 = 15 + 1*sin(t(i));
    sigma_x_0 = 5 + 1*sin(t(i));
    sigma_z_0 = 5 + 1*sin(t(i));
    
    mu_x_1 = 15 + 1*sin(t(i));
    mu_z_1 = -15 + 1*sin(t(i));
    sigma_x_1 = 5 + 1*sin(t(i));
    sigma_z_1 = 5 + 1*sin(t(i));
    
    mu_x_2 = 11 + 1*sin(t(i));
    mu_z_2 = 12 + 6*sin(t(i));
    sigma_x_2 = 13 + 9*sin(t(i));
    sigma_z_2 = 2 -6*sin(t(i));
    
    mu_x_3 = 10 -3*sin(t(i));
    mu_z_3 = 6 -5*sin(t(i));
    sigma_x_3 = 20 -4*sin(t(i));
    sigma_z_3 = 6 -1*sin(t(i));
    
    gauss =         gauss2D(X, Z, mu_x_0, mu_z_0, sigma_x_0, sigma_z_0, A0);
    gauss = gauss + gauss2D(X, Z, mu_x_1, mu_z_1, sigma_x_1, sigma_z_1, A1);
%     gauss = gauss + gauss2D(X, Z, mu_x_2, mu_z_2, sigma_x_2, sigma_z_2, A2);
%     gauss = gauss + gauss2D(X, Z, mu_x_3, mu_z_3, sigma_x_3, sigma_z_3, A3);
    
    [Gx,Gz] = imgradientxy(gauss);
    
    
    
    
    figure(1);
    subplot(1,3,1)
    imagesc(x, z, gauss)
    colorbar()
    title("gaussian")
    
    subplot(1,3,2)
    imagesc(x, z, Gx)
    title("Gx")
    colorbar()
    
    subplot(1,3,3)
    imagesc(x, z, Gz)
    title("Gz")
    colorbar()
end

% -------------------------------------------------------------------------
function gauss = gauss2D(X,Z,mux, muz, sigmax, sigmaz, A)
    gauss = A * exp(-((X - mux).^2 / (2*sigmax^2) + (Z - muz).^2 / (2*sigmaz^2)));
end