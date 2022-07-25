function [files]=fct_list_ext_files(path, ext, more)
    % Return the .mat file and removes all other types. For example, if a
    % folder contains a.mat, a.m, a.png the function returns a.mat
    
    listing=dir(fullfile(path, more, strcat('*.', ext)));
    for k=1:length(listing)
        files{k}=listing(k).name;
    end
        
end
