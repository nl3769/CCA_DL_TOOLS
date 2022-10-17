close all;
clearvars;

sample = 1000;
t = linspace(0, 2*pi, sample); 

sin = sin(t);
hann = squeeze(hanning(sample))';
RI = hann .* sin;

in_wave     = conv(RI, sin);
out_wave    = conv(RI, in_wave);


figure()
plot(sin)
title('sin')
figure()
plot(RI)
title('RI')
figure()
plot(in_wave)
title('in wave')
figure()
plot(out_wave)
title('out wave')

%plot(out_wave)

