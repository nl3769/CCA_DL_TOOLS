function [apod_window]=fct_get_apod_SL(dz, pitch, nb_sample, fnumber, Nactive, method)
      
    if method == 'adaptative'
        apod_window = adaptative(fnumber, pitch, dz, Nactive, nb_sample);
    end
    
end

function [apod_window] = adaptative(fnumber, pitch, dz, Nactive, nb_sample)
    %     % Returns a matrix containing the applied delay as a function of depth.
        
    % --- create Inf matrix
    apod_window = zeros([nb_sample Nactive]);
    
    coef = 10;
    D = (Nactive-1)*pitch;
    x_org=linspace(-D/2, D/2, Nactive);
    
    pitch_up=D/(coef-1);
    
    if fnumber == 0 fnumber=eps ; end
    
    
%     for id_delay=1:1:id_tresh-1
    for t_sample=1:nb_sample
        
        % --- find active element to preserve the fnumber
        Nactive_= round(t_sample*dz / (fnumber*pitch_up)+1);
%         disp(['Nactive: ' num2str(Nactive_)]);
        
        if Nactive_ > Nactive*coef Nactive_ = Nactive*coef; end
        
        if Nactive_ < 3
            Nactive_ = 3; 
        end               
        
        % -- get corresponding delay and apodization window
        apod_method = 'hanning';
        switch apod_method
            case 'hanning'
                apod_ = hanning(Nactive_);
                apod_ = apod_/sum(apod_);
            case 'rect'
                apod_ = ones(Nactive_, 1);
                apod_ = apod_/sum(apod_);
        end
        
        % -- store in Tx_delay_matrix
        x_interp = 0:pitch_up:(Nactive_-1)*pitch_up;
        width_ = (Nactive_ - 1)*pitch_up;
        x_interp = x_interp - width_/2;

        apod_window(t_sample, :) = interp1(x_interp, apod_', x_org, 'linear', 0)';
        apod_window(t_sample, :) =  apod_window(t_sample, :)/sum(apod_window(t_sample, :));
    end
    
end
