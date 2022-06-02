function [raw_data_name]=fct_get_raw_from_folder(list_files)
    

    inc=1;
    for k=1:size(list_files, 2)
        if contains(list_files{k}, 'raw_data')
            raw_data_name{inc}=list_files{k};
            inc=inc+1;
        end
    end
    
end