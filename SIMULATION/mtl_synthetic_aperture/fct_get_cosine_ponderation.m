function [mask] = fct_get_cosine_ponderation(height, width, dz, pitch)
    % Return cosine ponderation. It stores value in 3D matrix. 

    mask = zeros([height width width]); % 3D matrix
    
    % --- get starting and ending points
    for id_tx=1:width
        for id_rx=1:width
            for y=1:1:height
                
                % --- compute cosine value
                OPP = abs(id_tx*pitch - id_rx*pitch);
                ADJ = y*dz;
                cosine = ADJ / sqrt(ADJ^2 + OPP^2);
                
                % --- get corresponding time
                mask(y, id_rx, id_tx) = cosine;
                
            end
        end
    end

end