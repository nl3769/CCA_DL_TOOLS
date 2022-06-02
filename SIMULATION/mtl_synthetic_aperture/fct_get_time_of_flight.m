function [time_of_flight]=fct_get_time_of_flight(height, width, dz, pitch, c)
    % Return time of flight. It stores value in 3D matrix. 

    time_of_flight = zeros([height width width]); % 3D matrix
    
    % --- get starting and ending points
    for id_tx=1:width
        for id_rx=1:width
            for y=1:1:height
                
                % --- position
                pos_transducer = [id_tx*pitch, 0];
                pos_pixel = [id_rx*pitch, y*dz];
                
                % --- get corresponding time
                time_of_flight(y, id_rx, id_tx) = norm((pos_pixel-pos_transducer), 2)/c;
            
            end
        end
    end

end