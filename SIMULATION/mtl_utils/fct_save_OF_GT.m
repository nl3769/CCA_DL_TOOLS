function [] = save_OF_GT(OF, pres)
  % save optical flow in nifti format
  
  niftiwrite(OF, strcat(pres, '.nii'));

end
