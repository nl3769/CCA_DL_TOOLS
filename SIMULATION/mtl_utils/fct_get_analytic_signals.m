function [anal_sig] = fct_get_analytic_signals(RF_aperture, fs, fc)
	% Return analytic signal of the RF signal.
    
    anal_sig = zeros(size(RF_aperture));
    
    for emit=1:1:size(RF_aperture, 3)
        anal_sig(:,:,emit) = hilbert(RF_aperture(:,:,emit));
    end
    
end