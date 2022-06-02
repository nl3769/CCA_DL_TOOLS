function I_out = fct_expand_histogram(I_in, min_val, max_val)
    
    I_out = I_in - min(I_in(:));
    coef = max_val/double(max(I_out(:)));
    I_out = I_out * coef;
    I_out = I_out + min_val;

end