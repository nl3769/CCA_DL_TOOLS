function [RF_padding, tstart, tcompensation] = fct_zero_padding_RF_signals_DA(RF)
    
    RF_padding = RF;
    tstart = zeros(1, size(RF, 2));
    tcompensation = RF{1}(2);       % time compensation is the same for each column.
    
%     % --- we add zeros padding in order to get the correct RF signal
%     for id_t=1:1:size(tstart, 2)
%         nb_line_padding = (tstart(id_t) - tcompensation)/2*c/dz;
%         padding=zeros(nb_line_padding, 1);
%         RF_padding{id_apert}=[padding, RF{id_t}(3:end)];
%         a=1
%     end
    
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
        RF_padding{id_apert}=[RF{id_apert}(3:end) ; padding]; % get received signal from field simulation
	end
    
end