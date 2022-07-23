function [ndname] = remove_extension(dname)
    % Remove extension of a filename.

    ndname = erase(dname, ".tiff");
    ndname = erase(ndname, ".png");
    ndname = erase(ndname, ".jpg");
    ndname = erase(ndname, ".JPEG");
    ndname = erase(ndname, ".DICOM");
end
