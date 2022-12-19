function [Dz, Dx] = fct_get_gt(id_x, id_z, I)
    
    shape = size(id_x);

    % --- indexes computed by the methods
    idx_x = id_x(:);
    idx_z = id_z(:);

    % --- get corresponding GT displacment (
    motion_x = I(:, :, 1);
    motion_z = I(:, :, 3);
    
    % --- compute coordinates and extract them from GT
    idx = sub2ind(size(motion_x), idx_z, idx_x);
    Dx = motion_x(idx);
    Dz = motion_z(idx);
    
    % --- reshape
    Dx = reshape(Dx, shape);
    Dz = reshape(Dz, shape);

end