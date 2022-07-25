close all;
clearvars;


% --- add path
addpath(fullfile('..', 'mtl_cores/'));
addpath(fullfile('..', 'mtl_utils'));


pres = '/home/laine/HDD/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER';
seq = list_files(pres);

for pid=1:1:length(seq)
    pseq = fullfile(pres, seq{pid});
    
    id_seq = list_files(pseq);
    
    for idi=1:1:length(id_seq)
        pres_ = fullfile(pseq, id_seq{idi}, 'parameters');
        
        fname_mat = fct_list_ext_files(pres_, 'mat', '');
        fname_json_ = fct_list_ext_files(pres_, 'json', '');
        
        pres_mat = fullfile(pseq, id_seq{idi}, 'parameters', fname_mat{1});
        pres_json = fullfile(pseq, id_seq{idi}, 'parameters', fname_json_{1});
        
        param = load(pres_mat);
        param = param .p;
        
        delete(pres_json)
        delete(pres_mat)
        
        str = jsonencode(param);
        fid = fopen(pres_json, 'w');
        fprintf(fid, str); 
        fclose(fid);
        
    end
    
end

% -------------------------------------------------------------------------
function [files] = list_files(path)

    listing = dir(path);
    incr=1;
    
    for id=1:1:length(listing)
        if listing(id).name ~= '.'
            files{incr} = listing(id).name;
            incr=incr+1;
        end
    end

end