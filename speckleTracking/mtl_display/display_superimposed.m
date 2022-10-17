function display_superimposed(Dx, Dz, Dx_gt, Dz_gt, id_x, id_z, name)

        figure
        plot(id_z(1:2:end,1:2:end),id_x(1:2:end,1:2:end),'ko',...
            'MarkerFaceColor','k','MarkerSize',4)
        hold on
        h1 = quiver(id_z(1:2:end,1:2:end),id_x(1:2:end,1:2:end),...
            Dz(1:2:end,1:2:end),Dx(1:2:end,1:2:end),2, 'color',[1 0 0]);
        hold on
        h2 = quiver(id_z(1:2:end,1:2:end),id_x(1:2:end,1:2:end),...
            Dz_gt(1:2:end,1:2:end),Dx_gt(1:2:end,1:2:end),2, 'color',[0 1 0]);
        hold off
        
        set(h1, 'LineWidth', 1.5)
        set(h2, 'LineWidth', 1.5)
        axis equal ij tight
        title(name)
end