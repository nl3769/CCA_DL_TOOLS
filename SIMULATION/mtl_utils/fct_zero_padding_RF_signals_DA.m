function [RF_padding, tstart, tcompensation] = fct_zero_padding_RF_signals_DA(RF)
    
    RF_padding = RF;
    tstart = zeros(1, size(RF, 2));
    tcompensation = RF{1}(2);
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
        tstart(id_apert) = RF{id_apert}(1);
        RF_padding{id_apert}=[RF{id_apert}(3:end) ; padding];
	end
    
end