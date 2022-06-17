% function [time_of_flight]=fct_get_time_of_flight(height, width, dz, pitch, c)
    
function [tof]=fct_get_time_of_flight(X_img, Z_img, probe_pos_x, c)
% Return time of flight. It stores value in 3D matrix. 
    
    nb_event = size(probe_pos_x, 2);
    tof = zeros([size(X_img) nb_event]);
    
    for id=1:1:nb_event
       tof(:,:,id) = sqrt((probe_pos_x(id)-X_img).^2 + (Z_img).^2 ) ./ c; 
    end

end