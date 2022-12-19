function [path]=fct_get_path_from_substr(path, sub_str, list_files)
    

    for k=1:size(list_files, 1)
        if contains(list_files(k).name, sub_str) & ~contains(list_files(k).name, "swp")
            name = list_files(k).name;
            break;
        end
    end
    
    path = fullfile(path, name);
    
end