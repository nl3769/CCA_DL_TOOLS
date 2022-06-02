close all; clearvars ;

addpath('../MUST_2021/')

probe=getparam('L12-3v');
% probe.bandwidth=199;
[pulse, t]=getpulse(probe);
figure
plot(pulse);


% 
% % PULSE SPECTRUM
% mysinc = @(x) sinc(x/pi); % cardinal sine
% 
% 
% % We want a windowed sine of width PARAM.TXnow
% T = 1/probe.fc; % temporal pulse width
% wc = 2*pi*probe.fc;
% pulseSpectrum = @(w) 1i*(mysinc(T*(w-wc)/2)-mysinc(T*(w+wc)/2));
% 
% w=wc:1/10:wc;
% figure
% plot(pulseSpectrum(w));


function [pulse,t] = getpulse_local(param,way)

narginchk(1,2)
nargoutchk(1,2)
if nargin==1, way = 1; end % one-way is the default
assert(way==1 || way==2,'WAY must be 1 (one-way) or 2 (two-way)')

%-- Center frequency (in Hz)
assert(isfield(param,'fc'),...
    'A center frequency value (PARAM.fc) is required.')
fc = param.fc; % central frequency (Hz)

%-- Fractional bandwidth at -6dB (in %)
if ~isfield(param,'bandwidth')
    param.bandwidth = 75;
end
assert(param.bandwidth>0 && param.bandwidth<200,...
    'The fractional bandwidth at -6 dB (PARAM.bandwidth, in %) must be in ]0,200[')

%-- TX pulse: Number of wavelengths
if ~isfield(param,'TXnow')
    param.TXnow = 1;
end
NoW = param.TXnow;
assert(isscalar(NoW) && isnumeric(NoW) && NoW>0,...
    'PARAM.TXnow must be a positive scalar.')

%-- TX pulse: Frequency sweep for a linear chirp
if ~isfield(param,'TXfreqsweep') || isinf(NoW)
    param.TXfreqsweep = [];
end
FreqSweep = param.TXfreqsweep;
assert(isempty(FreqSweep) ||...
    (isscalar(FreqSweep) && isnumeric(FreqSweep) && FreqSweep>0),...
    'PARAM.TXfreqsweep must be empty (windowed sine) or a positive scalar (linear chirp).')

% MODIFIER ICI
mysinc = @(x) sinc(x/pi); % cardinal sine
% mysinc = @(x) double(x==0); % cardinal sine

% [note: In MATLAB, sinc is sin(pi*x)/(pi*x)]

%-- FREQUENCY SPECTRUM of the transmitted pulse
if isempty(FreqSweep)
    % We want a windowed sine of width PARAM.TXnow
    T = NoW/fc; % temporal pulse width
    wc = 2*pi*fc;
    pulseSpectrum = @(w) 1i*(mysinc(T*(w-wc)/2)-mysinc(T*(w+wc)/2));
else
    % We want a linear chirp of width PARAM.TXnow
    % (https://en.wikipedia.org/wiki/Chirp_spectrum#Linear_chirp)
    T = NoW/fc; % temporal pulse width
    wc = 2*pi*fc;
    dw = 2*pi*FreqSweep;
    s2 = @(w) sqrt(pi*T/dw)*exp(-1i*(w-wc).^2*T/2/dw).*...
        (fresnelint((dw/2+w-wc)/sqrt(pi*dw/T)) +...
        fresnelint((dw/2-w+wc)/sqrt(pi*dw/T)));
    pulseSpectrum = @(w) (1i*s2(w)-1i*s2(-w))/T;
end

%-- FREQUENCY RESPONSE of the ensemble PZT + probe
% We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
% (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
wB = param.bandwidth*wc/100; % angular frequency bandwidth
p = log(126)/log(2*wc/wB); % p adjusts the shape
probeSpectrum = @(w) exp(-(abs(w-wc)/(wB/2/log(2)^(1/p))).^p);
% The frequency response is a pulse-echo (transmit + receive) response. A
% square root is thus required when calculating the pressure field:
probeSpectrum = @(w) sqrt(probeSpectrum(w));
% Note: The spectrum of the pulse (pulseSpectrum) will be then multiplied
% by the frequency-domain tapering window of the transducer (probeSpectrum)

%-- frequency samples
dt = 1e-9; % time step is 1 ns
df = param.fc/param.TXnow/32;
p = nextpow2(1/dt/2/df);
Nf = 2^p;
f = linspace(0,1/dt/2,Nf);

%-- spectrum of the pulse
F = pulseSpectrum(2*pi*f).*probeSpectrum(2*pi*f).^way;

%-- pulse in the temporal domain (step = 1 ns)
tmp = [F conj(F(end-1:-1:2))];
pulse = fftshift(ifft(tmp,'symmetric'));
pulse = pulse/max(abs(pulse));

%-- keep the significant magnitudes
idx1 = find(pulse>(1/1023),1);
idx2 = find(pulse>(1/1023),1,'last');
idx = min(idx1,2*Nf-1-idx2);
pulse = pulse(end-idx+1:-1:idx);

%-- time vector
if nargout==2
    t = (0:length(pulse)-1)*dt;
end

% --- NL FOR DISPLAY

%%% pulseSpectrum_ in temporal domain
figure(1)
pulseSpectrum_=pulseSpectrum(2*pi*f);
tmp_ = [pulseSpectrum_ conj(F(end-1:-1:2))];
pulseSpectrum_ = fftshift(ifft(tmp_,'symmetric'));
pulseSpectrum_ = pulseSpectrum_/max(abs(pulseSpectrum_));
%-- keep the significant magnitudes
idx1 = find(pulseSpectrum_>(1/1023),1);
idx2 = find(pulseSpectrum_>(1/1023),1,'last');
idx = min(idx1,2*Nf-1-idx2);
pulseSpectrum_ = pulseSpectrum_(end-idx+1:-1:idx);
t_=(0:length(pulseSpectrum_)-1)*dt;
plot(t_, pulseSpectrum_)
title('pulse temporal domain')
xlabel('time in \mu s', 'FontSize', 20)
set(gca,'FontSize',20)

%%% pulseSpectrum_ in frequency domain
figure(1)
pulseSpectrum_=pulseSpectrum(2*pi*f);
tmp_ = [pulseSpectrum_ conj(F(end-1:-1:2))];
pulseSpectrum_ = fftshift(ifft(tmp_,'symmetric'));
pulseSpectrum_ = pulseSpectrum_/max(abs(pulseSpectrum_));
%-- keep the significant magnitudes
idx1 = find(pulseSpectrum_>(1/1023),1);
idx2 = find(pulseSpectrum_>(1/1023),1,'last');
idx = min(idx1,2*Nf-1-idx2);
pulseSpectrum_ = pulseSpectrum_(end-idx+1:-1:idx);
t_=(0:length(pulseSpectrum_)-1)*dt;
plot(t_, pulseSpectrum_)
title('pulse temporal domain')
xlabel('time in \mu s', 'FontSize', 20)
set(gca,'FontSize',20)

%%% probeSpectrum_ in temporal domain
figure(3)
probeSpectrum_=probeSpectrum(2*pi*f);
tmp_ = [probeSpectrum_ conj(F(end-1:-1:2))];
probeSpectrum_ = fftshift(ifft(tmp_,'symmetric'));
probeSpectrum_ = probeSpectrum_/max(abs(probeSpectrum_));
%-- keep the significant magnitudes
idx1 = find(probeSpectrum_>(1/1023),1);
idx2 = find(probeSpectrum_>(1/1023),1,'last');
idx = min(idx1,2*Nf-1-idx2);
probeSpectrum_ = probeSpectrum_(end-idx+1:-1:idx);
t__=(0:length(probeSpectrum_)-1)*dt;
plot(t__, probeSpectrum_)
title('probe temporal domain')
xlabel('time in \mu s', 'FontSize', 20)
set(gca,'FontSize',20)


%%% probeSpectrum_ in frequency domain
figure(4)
f_=1e6:df:15e6;
probeSpectrum_f=probeSpectrum(2*pi*f_);
% tmp_ = [probeSpectrum_f conj(F(end-1:-1:2))];
probeSpectrum_f = probeSpectrum_f/max(abs(probeSpectrum_f));
%-- keep the significant magnitudes
% idx1 = find(probeSpectrum_f>(1/1023),1);
% idx2 = find(probeSpectrum_f>(1/1023),1,'last');
% idx = min(idx1,2*Nf-1-idx2);
% probeSpectrum_f = probeSpectrum_f(end-idx+1:-1:idx);

plot(f_, probeSpectrum_f)
title('probe frequency domain')
xlabel('frequency in MHz', 'FontSize', 20)
set(gca,'FontSize',20)
xlim([1e6 15e6])

%%% transmitted pulse in frequency domain
figure(5)
f_=1e6:df:15e6;
transm_pulse_=pulseSpectrum(2*pi*f_).*probeSpectrum(2*pi*f_).^way;
pulseSpectrum_=pulseSpectrum(2*pi*f_);
probeSpectrum_=probeSpectrum(2*pi*f_).^way;
% tmp_ = [probeSpectrum_f conj(F(end-1:-1:2))];
% transm_pulse_ = transm_pulse_/max(abs(transm_pulse_));
%-- keep the significant magnitudes
% idx1 = find(probeSpectrum_f>(1/1023),1);
% idx2 = find(probeSpectrum_f>(1/1023),1,'last');
% idx = min(idx1,2*Nf-1-idx2);
% probeSpectrum_f = probeSpectrum_f(end-idx+1:-1:idx);

plot(f_, abs(transm_pulse_))
hold on
plot(f_, abs(pulseSpectrum_))
hold on
plot(f_, abs(probeSpectrum_))
title('transmitted pulse in frequency domain')
xlabel('frequency in MHz', 'FontSize', 20)
set(gca,'FontSize',20)
xlim([1e6 15e6])

end