classdef gaussian2DHandler < handle   
    
    properties (Access = private)
%         x_min;
%         x_max;
%         z_min;
%         z_max;
        time_sample;
    end
    
    properties
        
        nb_gaussian;  % 
        mu_x;         % (array) contain mu x value for each gaussian
        mu_z;         % (array) contain mu z value for each gaussian
        A;            % (array) contain amplitude for each gaussian
        sigma_x;      % (array) contain sigma x value for each gaussian
        sigma_z;      % (array) contain sigma z value for each gaussian
        
        mu_x_t; % (array) contain mu x value for each gaussian
        mu_z_t; % (array) contain mu z value for each gaussian
        A_t; % (array) contain amplitude for each gaussian
        sigma_x_t; % (array) contain sigma x value for each gaussian
        sigma_z_t; % (array) contain sigma z value for each gaussian
        
        coef_A; % store the new value of the amplitude for each gaussian
        coef_translation_x; % store the new mu value at each time sample for x dimension
        coef_translation_z; % store the new mu value at each time sample for z dimension
        coef_sigma_x; % store the new sigma value at each time sample for x dimension
        coef_sigma_z; % store the new sigma value at each time sample for z dimension
        
        current_gauss;  % store the gaussian at time sample t
        
        x_pts; % points in meter where the gaussian is define
        z_pts; % points in meter where the gaussian is define
        x_min; % min lateral dimension
        x_max; % max lateral dimension
        z_min; % min axial dimension
        z_max; % max lateral dimension
        nb_pts; % number of points to define the gaussian
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj = gaussian2DHandler(nb_gaussian, x_min, x_max, z_min, z_max, time_sample)
            
            % Constructor
            
            % --- set min and max values
            A_max=1;
            A_min=0.01;
            
            obj.x_min = x_min * 2;
            obj.x_max = x_max * 2;
            obj.z_min = z_min - z_max/2; % because z is always positive
            obj.z_max = z_max + z_max/2;
            obj.nb_pts =  50;
            sigma_x_max=(obj.x_max-obj.x_min)*20/100;
            sigma_x_min=(obj.x_max-obj.x_min)*5/100;
            
            sigma_z_max=(obj.z_max-obj.z_min)*20/100;
            sigma_z_min=(obj.z_max-obj.z_min)*5/100;

%             sigma_x_max=(obj.x_max-obj.x_min)*5/100;
%             sigma_x_min=(obj.x_max-obj.x_min)*0.5/100;
%             
%             sigma_z_max=(obj.z_max-obj.z_min)*5/100;
%             sigma_z_min=(obj.z_max-obj.z_min)*0.5/100;
            
            mu_x_max=obj.x_max;
            mu_x_min=obj.x_min;
            
            mu_z_max=obj.z_max;
            mu_z_min=obj.z_min;
            
            for i=1:1:nb_gaussian
                obj.mu_x(i)=mu_x_min+(mu_x_max-mu_x_min)*rand(1);
                obj.mu_z(i)=mu_z_min+(mu_z_max-mu_z_min)*rand(1);
                obj.sigma_x(i)=sigma_x_min+(sigma_x_max-sigma_x_min)*rand(1);
                obj.sigma_z(i)=sigma_z_min+(sigma_z_max-sigma_z_min)*rand(1);
                obj.A(i)=A_min+(A_max-A_min)*rand(1,1)*sign(rand(1) -1);                
            end
            
            obj.nb_gaussian=nb_gaussian;
            obj.time_sample=time_sample;
            
            obj.mu_x_t=obj.mu_x;
            obj.mu_z_t=obj.mu_z;
            obj.sigma_x_t=obj.sigma_x;
            obj.sigma_z_t=obj.sigma_z;
            obj.A_t=obj.A;
            
            obj.coef_translation_x=mu_x_min+(mu_x_max-mu_x_min)*rand(1);
            obj.coef_translation_z=mu_z_min+(mu_z_max-mu_z_min)*rand(1);
            obj.coef_sigma_x=sigma_x_min+(sigma_x_max-sigma_x_min)*rand(1);
            obj.coef_sigma_z=sigma_z_min+(sigma_z_max-sigma_z_min)*rand(1);
            
            obj.x_pts = linspace(obj.x_min, obj.x_max, obj.nb_pts);
            obj.z_pts = linspace(obj.z_min, obj.z_max, obj.nb_pts);
            
        end
        
        % ------------------------------------------------------------------
        function display_df(obj, fig_nb)
            
            gauss=obj.mk_gaussian();
%             gauss=gauss/max(gauss(:));
            [Gx, Gz] = imgradientxy(gauss);
            Gx = Gx / max(abs(Gx(:)));
            Gz = Gz / max(abs(Gz(:)));

            figure(fig_nb)
            [X, Z] = meshgrid(obj.x_pts, obj.z_pts);
            qp1 = quiver(X,Z,Gx, Gz, 'AutoScale','on');
            a=1
        end
        
        % ------------------------------------------------------------------
        function gauss = mk_gaussian(obj)
            % Make gaussian. Gaussian is twice bigger than the image.
            
            [X, Z]=meshgrid(obj.x_pts, obj.z_pts);
            gauss = gauss2D(X, Z, obj.mu_x_t(1), obj.mu_z_t(1), obj.sigma_x_t(1), obj.sigma_z_t(1), obj.A_t(1));
            for i=2:1:obj.nb_gaussian
                gauss = gauss + gauss2D(X, Z, obj.mu_x_t(i), obj.mu_z_t(i), obj.sigma_x_t(i), obj.sigma_z_t(i), obj.A_t(i));
            end
            
        end
        
        % ------------------------------------------------------------------
        function get_current_gauss2D(obj, id_t, f)
           % Compute the current gaussian which is function of time. The
           % frequency has to be relatively high compare to the frequency
           % of the probe in order to pertube the displacement field.
            
            for i=1:1:obj.nb_gaussian 
                obj.mu_x_t(i)=obj.mu_x(i) + obj.coef_translation_x*sin(2*pi*f*obj.time_sample(id_t));
                obj.mu_z_t(i)=obj.mu_z(i) + obj.coef_translation_z*sin(2*pi*f*obj.time_sample(id_t));
                obj.A_t(i)=obj.A(i)*cos(2*pi*f*obj.time_sample(id_t)); %+ sin(2*pi*f*obj.time_sample(id_t))*obj.A(i);
                obj.sigma_x_t(i)=obj.sigma_x(i) + obj.coef_sigma_x*sin(2*pi*f*obj.time_sample(id_t));
                obj.sigma_z_t(i)=obj.sigma_z(i) + obj.coef_sigma_z*sin(2*pi*f*obj.time_sample(id_t));    
            end
            
            gauss=obj.mk_gaussian();
            obj.current_gauss = gauss;
        
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
        function [scatt] = add_gaussian_motion(obj, scatt, CF)
           % Compute gradient displaceent field Dx and Dz and apply it to
           % the scatterers position.
            
            [Dx, Dz] = obj.get_displacement(CF);
            [X, Z] = meshgrid(obj.x_pts, obj.z_pts);

            x_query = scatt.x_scatt;
            z_query = scatt.z_scatt;

            scatt_x = interp2(X,Z,Dx,x_query,z_query);
            scatt_z = interp2(X,Z,Dz,x_query,z_query);

            scatt.x_scatt = scatt.x_scatt + scatt_x;
            scatt.z_scatt = scatt.z_scatt + scatt_z;

        end
        
        % ------------------------------------------------------------------
    end
end

% --------------------------------------------------------------------------
function gauss = gauss2D(X,Z,mux, muz, sigmax, sigmaz, A)

    gauss = A * exp(-((X - mux).^2 / (2*sigmax^2) + (Z - muz).^2 / (2*sigmaz^2)));
    
end