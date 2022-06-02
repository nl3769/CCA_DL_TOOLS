function [emit, transd, transmit]=setspectrum(param)
    
    k=1000;
    % --- impulsionnal response of the transducer
    Tg=2/param.fc;                                      % signal g(t) duration [sec]
    tg=0:1/(k*param.fc):Tg;                                         % sampled time vector [sec]
    if(~rem(length(tg), 2)), tg=[tg tg(end)+Ts]; end    % even Ng
    Ng=length(tg);                                      % sampled time vector length [number of samples]
%     transd=sin(2*pi*param.fc*tg).*hanning(Ng)';              % set signal for transducer in temporal domain
%     transd=dirac(tg);              % set signal for transducer in temporal domain
    transd=zeros(Ng, 1);
    transd(1)=1;
    % --- excitation signal e(t) to the emission aperture
    Te=1/param.fc;                                  % signal duration [sec]
    te=0:1/(k*param.fc):Te;                                     % sampled time vector [sec]
    Ne=length(te);                                  % sampled time vector length [number of samples]
%     emit=sin(2*pi*param.fc*te).*hanning(Ne)';          % excitation signal
    emit=sin(2*pi*param.fc*te);          % excitation signal
    % --- compute transmitted signal
    transmit=conv(transd, emit);
    figure(1)
    plot(transmit)
    title('transmitted signal')
    
end