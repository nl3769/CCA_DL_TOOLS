% ------------------------------------------------------------------------------------------------------------------------------
function [Tx_delay_matrix, apod_window]=get_Tx_delay_matrix_scanline_based(dz, probe, height, coef, fnumber, focus, apod_method)
    % Returns a matrix containing the applied delay as a function of depth.
    if ~isdeployed
        addpath(fullfile('..', '/package_synthetic_aperture'))
    end
    % --- create Inf matrix
    Tx_delay_matrix = Inf([height probe.Nelements]);
    apod_window = zeros([height probe.Nelements]);
    
    width=(probe.Nelements-1)*probe.pitch;
    x_org=linspace(-width/2, width/2, probe.Nelements);
    
    probe_=probe;
    probe_.Nelements=probe.Nelements*coef;
    probe_.pitch=width/(probe_.Nelements-1);
    
    if fnumber==0 fnumber=eps ; end
    fnumber=focus/((probe.Nelements-1)*probe.pitch);
    
    
    for id_delay=1:height
        
        % --- find active element to preserve the fnumber
        Nactive_=round(id_delay*dz/(fnumber*probe_.pitch)+1);
        if Nactive_ > probe.Nelements*coef Nactive_=probe.Nelements*coef; end
        if Nactive_ == 1 Nactive_=2; end
                Nactive_=probe.Nelements*coef;
        probe_.Nelements=Nactive_;
        
        % -- get corresponding delay and apodization window
        switch apod_method
            case 'hanning'
                apod_ = hanning(Nactive_);
            case 'rect'
                apod_ = ones(Nactive_, 1);
        end
        
        Tx_ = txdelay(0, focus, probe_);
        % -- store in Tx_delay_matrix
        x_interp = 0:probe_.pitch:(Nactive_-1)*probe_.pitch;
        width_ = (Nactive_ - 1)*probe_.pitch;
        x_interp = x_interp - width_/2;
        Tx_delay_matrix(id_delay, :) = txdelay(0, focus, probe);
        apod_window(id_delay, :) = interp1(x_interp, apod_, x_org, 'linear', 0);
        sum_ = sum(apod_window(id_delay, :));
        apod_window(id_delay, :) = apod_window(id_delay, :)/sum_;
    end
    
end