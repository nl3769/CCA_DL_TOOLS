function str = fct_build_path(tab, pos)
    % Build with a structure
    str_='/';
    for i=2:1:(size(tab, 1)-pos)
        str_=fullfile(str_, tab{i});
    end

    str=str_;
end