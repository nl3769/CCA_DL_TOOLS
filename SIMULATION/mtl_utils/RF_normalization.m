function RF_in = RF_normalization(RF_in)
    
    dim = size(RF_in, 2);
    max_arr = zeros(dim, 1);
    
    for i=1:1:dim
        max_arr(i) = max(RF_in{i}(:));
    end
    
    coef = max(max_arr);
    
    for i=1:1:dim
        RF_in{i} = RF_in{i}/coef;
    end
    
end