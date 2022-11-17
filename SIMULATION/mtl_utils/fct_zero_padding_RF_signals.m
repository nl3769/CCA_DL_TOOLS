function [RF_out] = fct_zero_padding_RF_signals(RF)
    
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
    
    RF_out = zeros(size(RF_padding{1}, 1), size(RF_padding{1}, 2), Nelements);
	for id_apert=1:size(RF, 2)
        RF_out(:,:,id_apert) = RF_padding{id_apert};
    end
end