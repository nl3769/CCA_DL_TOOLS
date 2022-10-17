function display_motion(Dx, Dz, id_x, id_z, name)

        figure
        
        plot(id_z(1:2:end,1:2:end),id_x(1:2:end,1:2:end),'ko',...
            'MarkerFaceColor','k','MarkerSize',4)
        hold on

        h = quiver(id_z(1:2:end,1:2:end),id_x(1:2:end,1:2:end),...
            Dz(1:2:end,1:2:end),Dx(1:2:end,1:2:end),2);
        
        hold off
        set(h,'LineWidth',1.5)
        axis equal ij tight
        title(name)

end