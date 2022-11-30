function [low_res_img_rot] = fct_phase_rotation(low_res_img, probe, z_start)
	% Return analytic signal of the RF signal.
    
    low_res_img_rot = zeros(size(low_res_img));
    
    for emit=1:1:size(low_res_img, 3)
        %-- Time vector
        nl = size(low_res_img,1);
        t = (0:nl-1)'/probe.fs ; %+ 2 * z_start / probe.c ;
        low_res_img_rot(:,:,emit) = rotation(low_res_img(:,:,emit), t, probe.fc);
    end
    
end

% -------------------------------------------------------------------------
function [sig] = rotation(sig, t, fc)
    sig = sig.*exp(1i*2*pi*fc*t);
end