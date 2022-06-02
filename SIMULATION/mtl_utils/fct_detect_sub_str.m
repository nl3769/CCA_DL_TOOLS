function [names]=fct_detect_sub_str(list_files, sub_str)
    

    inc=1;
    for k=1:size(list_files, 2)
        if contains(list_files{k}, sub_str)
            names{inc}=list_files{k};
            inc=inc+1;
        end
    end
    
end