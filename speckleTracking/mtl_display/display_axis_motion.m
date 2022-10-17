function display_axis_motion(OF, axis, name)

    figure()
    imagesc(OF(:,:,axis))
    colormap('hot')
    colorbar
    title(name)

end