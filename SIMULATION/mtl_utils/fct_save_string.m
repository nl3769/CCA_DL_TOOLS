function []=fct_save_string(path_res, file_name, string)
    
    % --- save string in a specify directory
    % -> path_res: path to save the string
    % -> file_name: name of the file
    % -> string: string you want to save
    disp("------------")
    path=fullfile(path_res, [file_name '.txt'])
    disp("------------")
    fid = fopen(path,'w');
    fprintf(fid, string);
    fclose(fid);

end