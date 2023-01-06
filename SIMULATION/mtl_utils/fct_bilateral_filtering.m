function [ima_fil]=fct_bilateral_filtering(img, sigma_i, sigma_s, neighbour, iteration)
    % Apply bilateral filterer
    
    % --- get image dimension
    [YY, XX]=size(img);
    ima_fil=zeros(size(img));
    
    for k=1:1:iteration
        % --- update image
        if k>1, img=ima_fil; end
        
        % --- loop over x
        for x=1:1:XX
          % --- loop over y
            for y =1:1:YY
            A = 0;
            B = 0;
                % --- loop over x-neighbours
                for i = -neighbour:neighbour
                    % --- loop over y-neighbours
                  for j = -neighbour:neighbour
                    a = x + i;
                    b = y + j;
                    % --- check borders condition
                    if (a>0) && (a<XX+1) && (b>0) && (b<YY+1)
                      d_i = abs(img(y,x)-img(b,a));     
                      A = A + exp(-(cast(d_i, 'double')+eps)/(2*sigma_i)^2)*exp(-(((a-x)^2+(b-y)^2))/(2*sigma_s)^2)*img(b, a);
                      B = B + exp(-(cast(d_i, 'double')+eps)/(2*sigma_i)^2)*exp(-(((a-x)^2+(b-y)^2))/(2*sigma_s)^2);
                    end        
                  end
                end
            % --- update image
            ima_fil(y, x) = A / B;        
          end
        end
    end

end