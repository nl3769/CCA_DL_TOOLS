function OF_out = fct_adapt_flow(I, flow, CF, z_start)
    
    [height_I, width_I] = size(I);
    [height_OF_, width_OF_, ~] = size(flow);

    width_roi = linspace(-width_I/2*CF, width_I/2*CF, width_I);
%     height_roi = linspace(0, height_I*CF, height_I) + z_start;
    height_roi = linspace(z_start, height_I*CF+z_start, height_I);
    width_OF = linspace(-width_OF_/2 * CF, width_OF_/2 * CF, width_OF_);
    height_OF = linspace(0, height_OF_ * CF, height_OF_); 
    
    [X,Y]   = meshgrid(width_OF, height_OF);
    [Xq,Yq] = meshgrid(width_roi ,height_roi);
    
    OF_out = zeros(height_I, width_I, 3);
    OF_out(:,:,1) = interp2(X, Y, flow(:,:,1), Xq, Yq);
    OF_out(:,:,2) = interp2(X, Y, flow(:,:,2), Xq, Yq);
    OF_out(:,:,3) = interp2(X, Y, flow(:,:,3), Xq, Yq);
    
end