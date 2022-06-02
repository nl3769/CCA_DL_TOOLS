function [out] = fct_interpolation(in, X, Z, Xq, Zq)
	
    [heigh, width] = size(Xq);
    [~, ~, depth] = size(in) ;
    out = zeros(heigh, width, depth);
    
    for i=1:depth
    	out(:,:,i) = interp2(X, Z, in(:,:,i), Xq, Zq, 'linear', 0);
    end
    
end