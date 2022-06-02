function [anal_sig] = fct_get_analytic_signals(RF_aperture)
	% Return analytic signal of the RF signal.
    
    for emit=1:1:size(RF_aperture, 2)
    	anal_sig{emit} = hilbert(RF_aperture{emit});
    end
    
end