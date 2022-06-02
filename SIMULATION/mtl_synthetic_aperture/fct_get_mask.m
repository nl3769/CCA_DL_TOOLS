function [mask_window] = fct_get_mask(dim, Nactive, pitch, name, f_number, dz)
    
    switch name
        
        case 'hanning_full'
            mask_window = mask_hanning_full(dim, Nactive);
        
        case 'hanning_adaptative'
            mask_window = mask_hanning_adaptative(dim, Nactive, pitch, f_number, dz);
            
        otherwise
                mask_window = ones(dim(2));
    end

end


% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------
% ---------------------------------------------------------------------------------------

function [mask_window] = mask_hanning_full(dim, Nactive)

    width = dim(2);
    height = dim(1);
    hanning_window=hanning(Nactive); % hanning window 
    [~, id_max]=max(hanning_window);
    mask_window = zeros([height width width]);
    mask_window_ = zeros([height width]);
    
    for id_emit=1:width

        mask_ = zeros([width, 1]);

        if (id_emit >= id_max) && (id_emit <= (width - id_max))
            start_ = id_emit - id_max+1;
            end_ = id_emit + id_max;
            mask_(start_:end_) = hanning_window;
        elseif id_emit < id_max
            mask_(1:Nactive) = hanning_window;
        elseif id_emit > (width - id_max)
            mask_(end-Nactive+1:end) = hanning_window;
        end
        
%         mask_window(: ,id_emit, :)= mask_;
        

        for y=1:height
            mask_window_(y, :) = mask_';
        end
%         figure(1)
%         imagesc(mask_window_)
        
        mask_window(: , id_emit, :) = mask_window_;
        
    end

end
% ---------------------------------------------------------------------------------------
function [mask_window] = mask_hanning_adaptative(dim, Nactive, pitch, f_number, dz)

    width = dim(2);
    height = dim(1);

    mask_window_ = zeros([height, Nactive]);
    mask_window = zeros([height, width, width]);
    x_probe = -(Nactive-1)*pitch/2:pitch:(Nactive-1)*pitch/2;

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
        mask_int = interp1(x_act_int, apod_act_int, x_int, 'linear', 0);

        mask = interp1(x_int, mask_int, x_probe);
        mask_window_(id_z, :) = mask;

    end          

    % --- window offset according to the element
    for id_tx=1:1:width

        if id_tx < Nactive/2
            id = 1;
        elseif id_tx >= (width - Nactive/2 +1)
            id = width - Nactive +1;
        else    
            id = id_tx - Nactive/2 +1;
        end

        mask_window(:, id:id+Nactive-1, id_tx) = mask_window_;

    end
end
