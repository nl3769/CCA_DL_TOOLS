function Iw = fct_warpped(I0, OF)

    I0 = double(I0);
    [row, col] = size(I0);
    x = linspace(1, col, col);
    z = linspace(1, row, row);
    [X_q, Z_q] = meshgrid(x, z);

    X_w = X_q + OF(:,:,1); 
    Z_w = Z_q + OF(:,:,3);

    Iw = griddata(X_w, Z_w, I0, X_q, Z_q);

end