function [] = fct_make_video(rpath, vname, loc, FR)

    % --- animation
    listing = dir(fullfile(rpath, '*.png'));
    incr=1;
    for id=1:1:length(listing)
        if contains(listing(id).name, loc)
            names{incr} = listing(id).name;
            incr=incr+1;
        end
    end
    names = sort(names);

    writerObj = VideoWriter(fullfile(rpath, [vname '.avi']));
    writerObj.FrameRate = FR;
    open(writerObj);
    for id=1:1:length(names)
        I = imread(fullfile(rpath, names{id}));
        writeVideo(writerObj, I);
    end
    close(writerObj);

end