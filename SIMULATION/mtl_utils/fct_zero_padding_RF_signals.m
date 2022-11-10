function [RF_padding] = fct_zero_padding_RF_signals(RF)
    
%     RF_padding = RF;
    
    % --- get height max
    height = 0;
    for id_apert=1:size(RF, 2)
    	if height < size(RF{id_apert}, 1) 
        	height=size(RF{id_apert}, 1); 
        end
    end
    
    % --- get number of emit transducer
    Nelements = size(RF{1}, 2);
    
	for id_apert=1:size(RF, 2)
    	nb_line_padding=height-size(RF{id_apert}, 1);
        padding=zeros(nb_line_padding, Nelements);
        RF_padding{id_apert}=[RF{id_apert}(2:end,:) ; padding];
	end
    
end