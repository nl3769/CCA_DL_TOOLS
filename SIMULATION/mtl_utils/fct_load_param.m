function [param]=fct_load_param(path_param, flag_cluster)
    
%     flag_cluster=true
    param=load(path_param);
    param=param.p;
%     if ~flag_cluster
%         param=load(path_param);
%         param=param.p;
%     else
% %         % --- modify path_res
%         path_res_local=path_param;
%         path_res_=split(path_res_local, '/');
%         path_res_cluster='/';
%         for id=2:1:size(path_res_, 1)-2
%             
%             if strcmp(path_res_{id}, 'nlaine') 
%                 path_res_{id}='laine';
%             elseif strcmp(path_res_{id}, 'cluster')
%                 path_res_{id}='';
%             end
%             
%             path_res_cluster=strcat(path_res_cluster, path_res_{id});
%             if id<size(path_res_, 1)
%                 path_res_cluster=strcat(path_res_cluster, '/');
%             end
%         end
%         disp(['path_res_cluster: ' path_res_cluster])
% % %             disp(['path_res_cluster: ' path_res_cluster]);
% %         path_res_cluster = replace(path_res_cluster, '//', '/');
% %         disp(['Load data: ' path_res_cluster])
%         param=load(erase(path_param, 'cluster'));
%         param = param.p;
%         param.path_res = path_res_cluster;
%         end
        

%         param.p.path_res=path_res_cluster;
%         param=param.p;
    
    end
%     
% end