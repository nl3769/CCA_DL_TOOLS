function [anal_sig] = fct_get_analytic_signals(RF_aperture, probe, t_offset)
	% Return analytic signal of the RF signal.
    
    anal_sig = zeros(size(RF_aperture));
    name = 'hilbert';
    for emit=1:1:size(RF_aperture, 3)
%         emit=90;
        
%         probe.t0 = 0;
        if strcmp(name, 'hilbert')
            anal_sig(:,:,emit) = hilbert(RF_aperture(:,:,emit));
        elseif strcmp(name, 'IQ')
            probe.t0 = t_offset(emit);
            anal_sig(:,:,emit) = rf2iq(RF_aperture(:,:,emit), probe);
            nl = size(RF_aperture(:,:,emit),1);
            t = (0:nl-1)'/probe.fs + probe.t0;
            anal_sig(:,:,emit) = anal_sig(:,:,emit).*exp(1i*2*pi*probe.fc*t);
        end
    end

end