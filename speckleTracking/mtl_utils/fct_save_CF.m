function [] = fct_save_CF(cf, path) 
    fid = fopen(fullfile(path, 'cf.txt'),'w');                                                                                                                            
    fprintf(fid, num2str(cf));                                                                                                                             
    fclose(fid); 
end