function substr = fct_get_substr_id_seq(id)

    if id < 10
        substr = ['00' num2str(id)];
    elseif id < 100
        substr = ['0' num2str(id)];
    else
        substr = num2str(id);
    end

end