classdef elasticDeformationHandler < handle   
    
    properties (Access = private)
        time_sample;
    end
    
    properties
        
        nb_init; % number of arrow to initialize the map 
        nb_pts; % number of points to define the displacement field
        A;      % amplitde of each initialization vectors
        A_max;  % (array) contain mu x value for each gaussian
        A_min;  % (array) contain mu z value for each gaussian
        
        t_sample;
        
        current_df; % store the displacement field at time sample t
        
        x_pts_init; % points in meter where the gaussian is define
        z_pts_init; % points in meter where the gaussian is define
        X_I;
        Z_I;
        x_val_init;
        z_val_init;
        x_pts;      % points in meter where the displacement field is define
        z_pts;      % points in meter where the displacement field is define
        x_min;      % min lateral dimension
        x_max;      % max lateral dimension
        z_min;      % min axial dimension
        z_max;      % max lateral dimension
        
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj = elasticDeformationHandler(x_min, x_max, z_min, z_max, t_sample, CF, magnitude_coef)
            
            % Constructor
            
            % --- set min and max values
            A_max = CF * magnitude_coef;
            A_min = -CF * magnitude_coef;
            
            obj.t_sample = t_sample;
            
            obj.x_min = x_min * 2;
            obj.x_max = x_max * 2;
            obj.z_min = z_min - z_max/2; % because z is always positive
            obj.z_max = z_max + z_max/2;
            
            obj.nb_init = 8;
            obj.nb_pts = 1000;
            obj.x_pts_init = linspace(obj.x_min, obj.x_max, obj.nb_init);
            obj.z_pts_init = linspace(obj.z_min, obj.z_max, obj.nb_init);
            
            obj.x_pts = linspace(obj.x_min, obj.x_max, obj.nb_pts);
            obj.z_pts = linspace(obj.z_min, obj.z_max, obj.nb_pts);
            [obj.X_I, obj.Z_I] = meshgrid(obj.x_pts_init, obj.z_pts_init);
            for i=1:1:size(obj.Z_I(:), 1)
                obj.x_val_init(i)=A_min+(A_max-A_min)*rand(1,1);
                obj.z_val_init(i)=A_min+(A_max-A_min)*rand(1,1);
            end
            obj.x_val_init = reshape(obj.x_val_init, size(obj.X_I));
            obj.z_val_init = reshape(obj.z_val_init, size(obj.Z_I));
        end
        
        % ------------------------------------------------------------------
        function display_df(obj, fig_nb)
            
            [x_df, z_df] = obj.mk_df();
            figure(fig_nb)
            imagesc(x_df)
            figure(fig_nb)
            imagesc(z_df)
            colorbar
            
            figure(fig_nb+1)
            [X, Z] = meshgrid(obj.x_pts, obj.z_pts);
            qp1 = quiver(X,Z,x_df,z_df, 'AutoScale','on');
            hold on
            qp2 = quiver(obj.X_I, obj.Z_I,obj.x_val_init*50,obj.z_val_init*50, 'AutoScale','off');
        
        end
        
        % ------------------------------------------------------------------
        function [x_df, z_df] = mk_df(obj)
            [X_q, Z_q] = meshgrid(obj.x_pts, obj.z_pts);
            
            x_df= interp2(obj.X_I, obj.Z_I, obj.x_val_init, X_q, Z_q, 'makima');
            z_df = interp2(obj.X_I, obj.Z_I, obj.z_val_init, X_q, Z_q, 'makima');
            
        end
                
        % ------------------------------------------------------------------
        function [Dx, Dz] = get_displacement(obj, CF)
            
            [Dx, Dz] = imgradientxy(obj.current_gauss);
            max_Dx = max(abs(Dx(:)));
            max_Dz = max(abs(Dz(:)));
            Dx = Dx / max_Dx * CF;
            Dz = Dz / max_Dz * CF;
            
        end
        
        % ------------------------------------------------------------------
        function [scatt] = add_elastic_motion(obj, scatt)
           % Compute gradient displaceent field Dx and Dz and apply it to
           % the scatterers position.
            
            [Dx, Dz] = obj.mk_df();
            [X, Z] = meshgrid(obj.x_pts, obj.z_pts);

            x_query = scatt.x_scatt;
            z_query = scatt.z_scatt;

            dx = interp2(X,Z,Dx,x_query,z_query);
            dz = interp2(X,Z,Dz,x_query,z_query);

            scatt.x_scatt = scatt.x_scatt + dx;
            scatt.z_scatt = scatt.z_scatt + dz;

        end
        
        % ------------------------------------------------------------------
    end
end
