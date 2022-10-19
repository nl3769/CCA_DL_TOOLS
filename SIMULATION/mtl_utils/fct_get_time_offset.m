function time_offset = fct_get_time_offset(RF)
    % Time offset is the first line of the save RF data. It has to be
    % removed to apply DAS algorithm
    time_offset = zeros(size(RF, 2), 1);
    for tx_id=1:1:size(RF, 2)
        time_offset(tx_id) = RF{tx_id}(1,1);
    end
    
end