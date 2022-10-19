function RF_in = RF_normalization(RF_in)
    
    dim = size(RF_in, 2);
    max_arr = zeros(dim, 1);
    
    for i=1:1:dim
        RF_in{i} = RF_in{i}(2:end,:);       % remove time offset
        max_arr(i) = max(RF_in{i}(:));      % get max of id_tx event
        if max_arr(i)==0
            max_arr(i) = 1;
        end
    end
        
    for i=1:1:dim
        RF_in{i} = RF_in{i}/max_arr(i);
    end

end