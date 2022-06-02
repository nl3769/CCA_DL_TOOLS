function [tx_delay_matrix] = fct_get_tx_delay(dz, nb_sample, Nactive, pitch, focus)
    
    % Returns a matrix containing the applied delay as a function of depth.
    
    
    % --- create Inf matrix
    tx_delay_matrix = Inf([nb_sample Nactive]);
    
    PARAM.pitch = pitch;
    PARAM.Nelements = Nactive;
    
    for t_sample=1:nb_sample
        
        % --- find active element to preserve the fnumber);
%         tx_delay_matrix(t_sample, :) = txdelay(0, dz*t_sample, PARAM);
        tx_delay_matrix(t_sample, :) = txdelay(0, focus, PARAM);
    end

end