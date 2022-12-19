function [Dz, Dx, id_z, id_x] = fct_compute_motion_GT(pairs)
    
    param           = [];
    param.winsize   = [15 15; 10 10; 5 5];              % size fo the region
    param.iminc     = 1;                                % image increment
    [Dz, Dx, id_z, id_x]   = sptrack(pairs,param);      % called function from MUST

end