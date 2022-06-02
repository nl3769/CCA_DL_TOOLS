function [out] = fct_padd_columns(in)
    
    max = get_max(in);
    out = add_zeros_padding(in, max);

end

function [max] = get_max(in)
    % Get maximal size of different signals
    max=0;
    dim = length(in);
    
    for id=1:dim
        if size(in{id}, 1)>max
            max = size(in{id}, 1);
        end
    end
    
end

function [out] = add_zeros_padding(in, max)
    % Add zeros padding at the end of RF signals
    dim = length(in);
    
    for id=1:1:dim
        
        diff = max - size(in{id}, 1);
                
        if diff > 0
            padding = zeros(diff, 1)
            out{id} = [in{id} ; padding]
        else
            out{id} = in{id};
        end
    end
    
end