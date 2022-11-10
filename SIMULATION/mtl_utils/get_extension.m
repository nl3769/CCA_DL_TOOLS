function [type] = get_extension(dname)
    % Get the extension of a filename.

    if contains(dname, '.tiff')
        type = 'TIFF';
    elseif contains(dname, '.png')
        type = 'PNG';
    elseif contains(dname, '.JPEG')
        type = 'JPEG';
    elseif contains(dname, '.jpg')
        type = 'JPEG';
    elseif contains(dname, '.mat')
        type = 'MAT';
    else
        type = 'DICOM';
    end

end
