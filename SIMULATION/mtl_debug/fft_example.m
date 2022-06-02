close all;
clearvars;


% Program to take the FFT of a pulse.
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures if you have the Image Processing Toolbox.
clear;  % Erase all existing variables. Or clearvars if you want.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 15;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amplitude = 2;
% numSamples = 1024;
% % Set up the time axis to go from 0 to 20 with 1000 sample points.
% t = linspace(0, 32, numSamples);
% % Create a signal in the time domain.
% timeBasedSignal = zeros(1, numSamples); % First make everything zero.
% % Now figure out what indexes go from t=2 to t=6
% pulseIndexes = (t >= 2) & (t <= 6); % Logical indexes.
% indexes=find(pulseIndexes==1);
% window=hanning(sum(pulseIndexes))';
% % Now make the pulse.
% 
% timeBasedSignal(min(indexes):max(indexes)) = window;
% % timeBasedSignal=hanning(numSamples)';
% % Plot time based signal
% subplot(2, 1, 1);
% plot(t, timeBasedSignal, 'b-', 'LineWidth', 2);
% grid on;
% xlabel('Time', 'FontSize', fontSize);
% ylabel('Signal Amplitude', 'FontSize', fontSize);
% title('Signal in the Time Domain', 'FontSize', fontSize);
% ylim([0, 3]); % Set range for y axis to be 0 to 3.
% % Enlarge figure to full screen.
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
% % Not take the Fourier Transform of it.
% ft = fft(timeBasedSignal);
% % Shift it so that the zero frequency signal is in the middle of the array.
% ftShifted = fftshift(ft);
% % Get the magnitude, phase, real part, and imaginary part.
% ftMag = fftshift(abs(ft));
% ftReal = fftshift(real(ft));
% ftImag = fftshift(imag(ft));
% ftPhase = ftImag ./ ftReal;
% % Compute the frequency axis (I'm a little rusty on this part so it might not be right).
% % freqs = linspace(-1/(2*min(t)), 1/(2*max(t)), numSamples);
% deltat = max(t) - min(t);
% freqs = linspace(-1/(2*deltat), 1/(2*deltat), numSamples);
% % Plot the magnitude of the spectrum in Fourier space
% subplot(2, 1, 2);
% plot(freqs, ftMag, 'b-', 'LineWidth', 2);
% grid on;
% xlabel('Frequency', 'FontSize', fontSize);
% ylabel('Magnitude', 'FontSize', fontSize);
% title('Magnitude of the Signal in the Frequency (Fourier) Domain', 'FontSize', fontSize);


fc=7.5e6;
fs=30*fc;
Ts=1/fs;

% --- TG
Tg=2/fc;                                        % signal g(t) duration [sec]
tg=0:Ts:Tg;                                     % sampled time vector [sec]
t=0:Ts:10*Tg;
time_based_signal_TG=zeros(length(t), 1);
Ng=length(tg);                                  % sampled time vector length [number of samples]
g = sin(2*pi*fc*tg).*hanning(Ng)';              % signal
time_based_signal_TG(2*round(Tg/Ts):round(3*Tg/Ts))=g;

% --- TE
Te=1/fc;                                        % signal g(t) duration [sec]
te=0:Ts:Te;                                     % sampled time vector [sec]
time_based_signal_TE=zeros(length(t), 1);
Ne=length(te);                                  % sampled time vector length [number of samples]
e = sin(2*pi*fc*te).*hanning(Ne)';              % signal
time_based_signal_TE(round(Te/Ts):round(2*Te/Ts))=e;

% --- CONVOLUTION
c=conv(time_based_signal_TE, time_based_signal_TG);
t_c=(0:size(c, 1)-1)/fc;

% FOURIER DOMAIN
f_te=fftshift(abs(fft(time_based_signal_TE)));
f_tg=fftshift(abs(fft(time_based_signal_TG)));
f_c=fftshift(abs(fft(c)));

% FOURIER DOMAIN MULTIPLICATION
f_c_times=fft(time_based_signal_TE).*fft(time_based_signal_TG);
f_c_times=fftshift(abs(f_c_times));

% --- PLOT FIGURE
figure(1)
subplot(2, 1, 1);
plot(t*1e6, time_based_signal_TG)
title('TG')
subplot(2, 1, 2);
f = (-length(t)/2:1:length(t)/2-1)*(fs/length(t));     
plot(f', f_tg);
title('FFT TG')

figure(2)
subplot(2, 1, 1);
plot(t*1e6, time_based_signal_TE)
title('TE')
subplot(2, 1, 2);
f = (-length(t)/2:1:length(t)/2-1)*(fs/length(t));     
plot(f', f_te);
title('FFT TE')

figure(3)
subplot(3, 1, 1);
plot(t_c*1e6, c)
title('TE*TC (convolution)')
subplot(3, 1, 2);
f = (-length(t_c)/2:1:length(t_c)/2-1)*(fs/length(t_c));    
plot(f, f_c)
title('FFT TE*TC')
subplot(3, 1, 3);
deltat = max(t) - min(t);
f = (-length(t)/2:1:length(t)/2-1)*(fs/length(t));    
plot(f, f_c_times)
title('FFT TE*TC (fourier products)')

