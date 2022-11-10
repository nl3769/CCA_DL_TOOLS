function [id_start, id_end] = fct_get_se(Nelement, pitch, Nactive, x_img)
    
    x = -(Nelement-1)*pitch/2:pitch:(Nelement-1)*pitch/2;
    
    x_start = min(x) + (Nactive-1)/2 * pitch;
    x_end = max(x) - (Nactive-1)/2 * pitch;
    
    x_end = find_closed_value(x_img, x_end);
    x_start = find_closed_value(x_img, x_start);
        
    id_start = find(x_img==x_start);
    id_end = find(x_img==x_end);
    
end

function [val] = find_closed_value(arr, in)
    
    val = 0;
    diff_ = Inf;
    for id=1:1:length(arr)
        diff = in - arr(id);
        
        if abs(diff) < abs(diff_)
            val = arr(id);
            diff_ = diff;
        end
        
    end
    
end