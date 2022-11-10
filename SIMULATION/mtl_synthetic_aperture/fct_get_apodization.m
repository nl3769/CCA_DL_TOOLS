function [apod_window] = fct_get_apodization(dim, Nactive, pitch, name, f_number, dz)
    
    switch name
        
        case 'hanning_full'
            apod_window = apod_hanning_full(dim, Nactive);
        
        case 'hanning_adaptative'
            apod_window = apod_hanning_adaptative(dim, Nactive, pitch, f_number, dz);
            
        otherwise
                apod_window = ones(dim(2));
    end

end


% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------

function [apod_window] = apod_hanning_full(dim, Nactive)

    width = dim(2);
    height = dim(1);

    apod_window = zeros([height, width, width]);
  
    apod = hanning(Nactive)';
    apod = apod / sum(apod);
    apod_window_ = repelem(apod, height, 1);

    % --- Window offset according to the element

    for id_tx=1:1:width

        if id_tx >=Nactive/2 && id_tx < (width - Nactive/2 +1)
            id = floor(width - Nactive +1);
            apod_window(:, id:id+Nactive-1, id_tx) = apod_window_;
        end


    end
end

% ---------------------------------------------------------------------------------------
function [apod_window] = apod_hanning_adaptative(dim, Nactive, pitch, f_number, dz)

    width = dim(2);
    height = dim(1);

    apod_window_ = zeros([height, Nactive]);
    apod_window = zeros([height, width, width]);
    x_probe = -(Nactive-1)*pitch/2:pitch:(Nactive-1)*pitch/2;
    
    % --- Create apodization window of size depth*Nelements
    for id_z=1:1:height
        z = id_z*dz;
        Nactive_ = round(z/(pitch*f_number));

        if Nactive_ < 3 
            Nactive_ = 3 ; 
        elseif Nactive_ >= Nactive 
            Nactive_ = Nactive;
        end

        apod_act_int = hanning(Nactive_*10)';
        x_act_int = -(Nactive_*10-1)*pitch/10/2 : pitch/10 : (Nactive_*10-1)*pitch/10/2;

        x_int = -(Nactive * 10 - 1) * pitch/10/2 : pitch/10 : (Nactive * 10 - 1) * pitch/10/2;
        apod_int = interp1(x_act_int, apod_act_int, x_int, 'linear', 0);

        apod = interp1(x_int, apod_int, x_probe);
        
        % --- sum has to equal equal to zero
        sum_ = sum(apod);
        
        apod_window_(id_z, :) = apod/sum_;

    end          

    % --- window offset according to the element
    for id_tx=1:1:width

        if id_tx > Nactive/2 && id_tx < (width - Nactive/2 +1)
            id = floor(id_tx - Nactive/2 +1);
            apod_window(:, id:id+Nactive-1, id_tx) = apod_window_;
        end

    end
    
end