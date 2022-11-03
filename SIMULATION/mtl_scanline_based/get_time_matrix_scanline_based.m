% ------------------------------------------------------------------------------------------------------------------------------
function [time_matrix_2D_px_to_rx]=get_time_matrix_scanline_based(height, width, dz, pitch, c, id_f)
    % Compute the time matrix at a specific x value.
    
    time_matrix_2D_px_to_rx=zeros([height width]);  % 2D matrix
    width_ph=(width-1)*pitch;                       % width of the aperture in meter
    x=linspace(-width_ph/2, width_ph/2, width);
    
    for line=1:1:height
        for id_tx=1:width
            
            % --- position
            pos_pixel = [id_f, (line-1)*dz];
            pos_tx = [x(id_tx), 0];

            % --- get corresponding time
            time_matrix_2D_px_to_rx(line, id_tx) = norm((pos_pixel-pos_tx), 2)/c;
            
        end
    end
    
end