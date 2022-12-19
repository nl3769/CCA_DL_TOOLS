function [] = fct_save_res(id_x, id_z, Dx, Dz, Dx_gt, Dz_gt, pres)
    
    % --- save prediction
    filename = fullfile(pres, 'pred.nii');
    V = cat(3, Dx, Dz);
    niftiwrite(V, filename);
    
    % --- save ground truth
    filename = fullfile(pres, 'ref.nii');
    V = cat(3, Dx_gt, Dz_gt);
    niftiwrite(V, filename);
    
    % ---- save grid
    filename = fullfile(pres, 'grid.nii');
    V = cat(3, id_x, id_z);
    niftiwrite(V, filename);
    
end