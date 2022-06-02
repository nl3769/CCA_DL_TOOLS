function [I] = fct_create_image(cols)
    
    width = length(cols);
    height = size(cols{1}, 1);
    
    I = zeros([height, width]);
    
    for col=1:1:width
       I(:,col) = cols{col}; 
    end
    
end