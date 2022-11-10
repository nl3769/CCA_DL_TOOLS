classdef imageReconstruction < handle 
   
    properties
        param;
    end
    
    properties (Access = private)
        RF_aperture;
        probe;
        sub_probe;
        phantom;
        tx_delay;
        tstart;
        tcompensation;
        apod;
        low_res_image;
        RF_final;
        IQ;
        bmode;
        image;
        path_res;
        x_display;
        z_display;
        in_vivo;
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj=imageReconstruction(varargin)
            % Constructor
            
            switch nargin
                case 6
                    % varargin{1} -> path_data
                    % varargin{2} -> name of the radio frequence data
                    % varargin{3} -> param_name
                    % varargin{4} -> probe name
                    % varargin{5} -> subprobe name
                    % varargin{6} -> phantom_name
                    
                    [obj.RF_aperture, obj.probe, obj.sub_probe, obj.param, obj.phantom]=fct_get_data(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6});
                    obj.tstart = zeros(size(obj.RF_aperture, 1), 65);
                    obj.tcompensation = 0;
                    %obj.param.path_res = '/home/laine/Desktop/STA_TEST_/tech_001/tech_001_id_001_FIELD_3D/';
                    path_image_information=fullfile(obj.param.path_res, 'phantom', 'image_information.mat');
                    image=load(path_image_information);
                    obj.image=image.image;
                    obj.path_res=fct_create_directory(varargin{1}, varargin{2});

                otherwise
                    disp('Problem with parameters (imageReconstruction constructor)');
            end
            
        end
        
        % ------------------------------------------------------------------
        function beamforming_scanline_based(obj)
            % Apply DAS algorithm

            disp('run das_one_focus_with_aperture_simus')
            
            obj.tx_delay=txdelay(0, obj.phantom.depth_of_focus, obj.sub_probe);
%             obj.apod=hanning(obj.sub_probe.Nelements)';
            obj.apod=ones(obj.sub_probe.Nelements, 1)';

            dz=obj.probe.c/(2*obj.probe.fs);
            z_das=0:dz:obj.phantom.z_max;
            x_das=zeros(1, size(z_das, 2));                                     
            obj.RF_final=zeros(size(z_das, 2), size(obj.RF_aperture, 2));       
            
            obj.sub_probe.fnumber=0;
            obj.probe.fnumber=0;
            for id=1:1:size(obj.RF_aperture, 2)
                
                disp(['DAS process: ' num2str(id)])

                RF_=obj.RF_aperture{id}(:, id:id+obj.param.Nactive-1);
                RF_=bsxfun(@times, obj.apod, RF_);
                obj.RF_final(:,id)=das(RF_, x_das, z_das, obj.tx_delay, obj.sub_probe, 'linear'); % MUST function
            end
                        
            % --- apply TGC
            if obj.param.TGC==true
                obj.RF_final=tgc(obj.RF_final);
            end

        end
        
        % ------------------------------------------------------------------
        function DAS_scanline_based(obj, name_algo)
            % Delay and Sum algorithm beamforming for scanline based acquisition.

            if ~isdeployed
                addpath(fullfile('..', 'mtl_scanline_based'))
            end
            % --- add zero padding to fix potential dimension error
            obj.RF_aperture = fct_zero_padding_RF_signals(obj.RF_aperture);
            height=size(obj.RF_aperture{1}, 1);
            % --- delta z
            dz=obj.param.c/(2*obj.sub_probe.fs);
            
            % --- create time matrix
            time=(0:1:height-1)'/obj.probe.fs;     
            
            % --- get time matrix from pixel to element
            time_matrix = get_time_matrix_scanline_based(height, obj.sub_probe.Nelements, dz, obj.sub_probe.pitch, obj.param.c, 0);
%             time_matrix = MEX_TOF_SLB(obj.sub_probe.pitch, obj.sub_probe.Nelements, height, dz, obj.param.c);
            % --- compute delay matrix and apodization coefficients
            Tx_delay_matrix = fct_get_tx_delay(dz, height, obj.param.Nactive, obj.probe.pitch, obj.phantom.depth_of_focus);
            apod_window = fct_get_apod_SL(dz, obj.probe.pitch, height, obj.param.fnumber, obj.param.Nactive, 'adaptative');
            
            % --- matrix in which we store reconstructed lines
            y_m = zeros(height, obj.sub_probe.Nelements^2);
            
            % --- beamforming
            for id_apert = 1:(obj.probe.Nelements-obj.sub_probe.Nelements+1)
                sig = obj.RF_aperture{id_apert}(:, id_apert:id_apert+obj.sub_probe.Nelements-1);
                inc = 1;
                for id_tx=1:1:obj.sub_probe.Nelements
                    for id_rx=1:1:obj.sub_probe.Nelements
                        time_ = time_matrix(:, id_tx) + time_matrix(:, id_rx) + Tx_delay_matrix(:, id_tx) ; %- max(Tx_delay_matrix(:, id_tx));
                        y_m(:, inc) = apod_window(:, id_rx).*interp1(time, sig(:, id_rx), time_, 'makima', 0);
                        inc = inc + 1;
                    end
                end
                
                if strcmp(name_algo, 'DAS')
                    obj.RF_final(:, id_apert)=sum(y_m, 2);
                    
                elseif strcmp(name_algo, 'DMAS')
                    
                    % --- preserve dimensionality
                    y_m_1=sign(y_m).*sqrt(abs(y_m));
                    
                    % --- compute yDMAS
                    col_high_image_1=sum(y_m_1, 2);
                    col_high_image_1=col_high_image_1.^2;
                    col_high_image_2=y_m_1.^2;
                    col_high_image_2=sum(col_high_image_2, 2);
                    
                    % --- store result
                    obj.RF_final(:, id_apert)=col_high_image_1-col_high_image_2;
                    
                elseif strcmp(name_algo, 'DMAS_CF')
                    
                    % --- preserve dimensionality
                    y_m_1=sign(y_m).*sqrt(abs(y_m));
                    
                    % --- compute yDMAS
                    col_high_image_1=sum(y_m_1, 2);
                    col_high_image_1=col_high_image_1.^2;
                    col_high_image_2=y_m_1.^2;
                    col_high_image_2=sum(col_high_image_2, 2);
                    
                    % --- store result
                    y_DMAS=col_high_image_1-col_high_image_2;
                    
                    % --- compute coherence factor
                    M=size(y_m, 1);
                    y_den=M*sum(y_m.^2, 2);
                    
                    % --- store result
                    obj.RF_final(:, id_apert)=abs(y_DMAS).^2./(abs(y_den)+eps);      
                end

                disp(['Line ' num2str(id_apert) ' done.'])
            
            end
            
        end
        
        % ------------------------------------------------------------------
        function scan_conversion(obj)
            % Convert the pixel size to the original one
            % --- vertical grid
            
            n_pixels_start = ceil(obj.param.remove_top_region / obj.image.CF);
            
            z_start = obj.param.remove_top_region;
            z_end = obj.image.height * obj.image.CF;
            n_points_z = obj.image.height - n_pixels_start;
            
            % --- horizontal grid
            dim_phantom = obj.phantom.x_max-obj.phantom.x_min;
            dim_RF = obj.probe.pitch * (size(obj.RF_final, 2)-1);
            
            if dim_RF > dim_phantom
                x_start = obj.phantom.x_min;
                x_end = obj.phantom.x_max;
                n_points_x = obj.image.width;
            else
                width = dim_RF;
                x_start = -width/2;
                x_end = width/2;
                n_points_x = round(width / obj.image.CF);
                k = 2;
                offset = k*obj.probe.pitch;
                x_start = x_start + offset;
                x_end = x_end - offset;
                n_points_x = n_points_x - 2*k;
            end

            xq = linspace(x_start, x_end, n_points_x);
            x_org=linspace(-dim_RF/2, dim_RF/2, size(obj.RF_final, 2));
            
            dz = obj.probe.c/(2*obj.probe.fs);
            zq=linspace(z_start, z_end, n_points_z);
            z_org=linspace(0, size(obj.RF_final, 1) * dz, size(obj.RF_final, 1));

            [Xq, Zq]=meshgrid(xq, zq);
            [X_org, Z_org]=meshgrid(x_org, z_org);

            % --- adapt signal (time compensation and negative tstart in
            % case of dynamic acquisition)
            t_compensation = ones(1, size(obj.tstart, 2)) * obj.tcompensation;
            Z_org=bsxfun(@plus, Z_org, (obj.tstart - t_compensation)/2*obj.param.c);

            F = scatteredInterpolant(X_org(:), Z_org(:), double(obj.bmode(:)));
            fimg = F(Xq(:), Zq(:));
            obj.bmode = obj.bmode / max(obj.bmode(:));
            obj.bmode = reshape(fimg, size(Xq)); 
            % --- perform a low pass filtering
            obj.bmode=imgaussfilt(obj.bmode, 1, 'FilterSize', 3, 'Padding', 'symmetric');
            % --- perform a median filtering
            obj.bmode = medfilt2(obj.bmode);
            % --- apply treshold
            obj.bmode=imadjust(obj.bmode, [obj.param.imadjust_vals(1) obj.param.imadjust_vals(2)], []);
            
            obj.bmode = fct_expand_histogram(obj.bmode, 0, 255);
            obj.x_display=[x_start, x_end];
            obj.z_display=[z_start, z_end]; 
            
        end
        
        % ------------------------------------------------------------------
        function get_bmode_gamma_correction(obj)
            % Get bmode image withtout applying log compression. Only gamma
            % correction is applyed.
            
            % --- RF to IQ
            obj.IQ=rf2iq(obj.RF_final, obj.probe);
            % --- apply TGC
            if obj.param.TGC==true
                obj.IQ=tgc(obj.IQ);
            end
            % --- compute envelope
            obj.bmode = abs(obj.IQ); % real envelope
            % --- apply gamma correction
            obj.bmode = obj.bmode.^(obj.param.gamma);
            % --- apply image adjustement
            obj.bmode = obj.bmode/max(obj.bmode(:));
            
            if ~obj.param.dynamic_focusing
                % --- perform a low pass filtering
                obj.bmode=imgaussfilt(obj.bmode, 1, 'FilterSize', 3, 'Padding', 'symmetric');
                % --- perform a median filtering
                obj.bmode = medfilt2(obj.bmode);
                % --- apply treshold
                obj.bmode=imadjust(obj.bmode, [obj.param.imadjust_vals(1) obj.param.imadjust_vals(2)], []);
            end
        end
        
        % ------------------------------------------------------------------
        function update_probe(obj, id)
            % Modifies probe according to the parameters
            
            obj.probe.fnumber=obj.param{id}.fnumber;
        end
        
        % ------------------------------------------------------------------
        function update_sub_probe(obj, id)
            % Modifies sub_probe according to the parameters.
            
            obj.sub_probe.fnumber=obj.param{id}.fnumber;
            obj.sub_probe.Nelements=obj.param{id}.Nactive;
        end
        
        % ------------------------------------------------------------------
        function save_beamformed_data(obj)
            % Save bmode image image with the dimension of the phantom

            % --- path to save the IQ signal
            rpath=fullfile(obj.path_res, [obj.param.phantom_name '_BF.png']);                                        

            % --- compute the dimension of the image
            dim=size(obj.bmode);
            x_disp=linspace(obj.x_display(1), obj.x_display(2), dim(2));
            z_disp=linspace(obj.z_display(1), obj.z_display(2), dim(1));
            
            % --- add depth of focus
            DF=obj.phantom.depth_of_focus-obj.param.shift;
            
            % --- save the image
            f=figure('visible', 'off');
            imagesc(x_disp*1e3, z_disp*1e3, abs(obj.IQ)); 
            hold on
            line([obj.x_display(1)*1e3, obj.x_display(1)*1e3*0.95], [DF*1e3, DF*1e3], 'Color', 'r', 'LineWidth', 1)
            line([obj.x_display(2)*1e3*0.95, obj.x_display(2)*1e3], [DF*1e3, DF*1e3], 'Color', 'r', 'LineWidth', 1)
            hold off 
            colormap gray;
            colorbar;
            title('BF signals')
            xlabel('width in mm')
            ylabel('height in mm')
            saveas(f, rpath);
            
            % --- save signals in .nii file
            IQ = obj.IQ;
            niftiwrite(IQ, fullfile(obj.path_res, 'IQ_data.nii'));
            
        end
        
        % ------------------------------------------------------------------
        function [psave, pres_sim] = save_bmode(obj, scatt)
            % Save bmode image image with the dimension of the phantom

            % --- we save the image
            pres_=fullfile(obj.path_res, [obj.param.phantom_name '_bmode_result_physical_dimension.png']);

            
            obj.bmode = fct_expand_histogram(obj.bmode, 0, 255);

            % --- compute the dimension of the image
            dim=size(obj.bmode);
            x_disp=linspace(obj.x_display(1), obj.x_display(2), dim(2));
            z_disp=linspace(obj.z_display(1), obj.z_display(2), dim(1));
            
            % --- add depth of focus
            DF=obj.phantom.depth_of_focus-obj.param.shift;
            
            % --- save the image
            f=figure('visible', 'off');
            imagesc(x_disp*1e3, z_disp*1e3, obj.bmode); 
            hold on
            if scatt
                plot(obj.phantom.x_scatt*1e3, (obj.phantom.z_scatt-obj.param.shift)*1e3, 'go','MarkerSize',10)
            end
            line([obj.x_display(1)*1e3,obj.x_display(1)*1e3*0.95],[DF*1e3, DF*1e3],'Color','r','LineWidth',1)
            line([obj.x_display(2)*1e3*0.95,obj.x_display(2)*1e3],[DF*1e3, DF*1e3],'Color','r','LineWidth',1)
            
            hold off 
            colormap gray;
            title('Bmode image')
            xlabel('width in mm')
            ylabel('height in mm')
            saveas(f, pres_);
            
            % --- we save the image
            pres_sim=fullfile(obj.path_res, [obj.param.phantom_name '_bmode.png']);

            imwrite(uint8(obj.bmode), pres_sim);
            psave = obj.path_res;

%             niftiwrite(obj.bmode, fullfile(obj.path_res, 'bmode_result', 'bmode_simulated.nii'));
%             niftiwrite(obj.bmode, fullfile(obj.path_res, 'bmode_result', 'bmode_simulated.nii'));
        end
        
        % ------------------------------------------------------------------
        function [X_image, Z_image, X_org, Z_org]=get_grids_synthetic_aperture(obj, dim)
            % Returns the corresponding grid to obtain isotropic pixels.            
            
            % --- bmode image dimension
            z_start=obj.param.shift;                                                        % m
            z_end=obj.phantom.z_max;                                                        % m
            n_points_z = obj.image.height;                                                    % nb of points
            dim_phantom=obj.phantom.x_max-obj.phantom.x_min;
            dim_probe=obj.probe.pitch*(obj.probe.Nelements-1);
            if dim_probe>dim_phantom
                delta=(dim_probe-dim_phantom)/2;
                x_start=-obj.probe.pitch*(obj.probe.Nelements-1)/2+delta;
                x_end=obj.probe.pitch*(obj.probe.Nelements-1)/2-delta;
                n_points_x=obj.image.width;
            else
                x_start=-obj.probe.pitch*(obj.probe.Nelements-1)/2;
                x_end=-x_start;
                n_points_x=round((x_end-x_start)/obj.image.CF);
            end
            
            
            x_image=linspace(x_start, x_end, n_points_x);
            x_org=linspace(-obj.probe.pitch*(obj.probe.Nelements-1)/2, obj.probe.pitch*(obj.probe.Nelements-1)/2, dim(2));
            
            z_image=linspace(z_start, z_end, n_points_z);
            z_org=linspace(0, dim(1)*obj.probe.c/(2*obj.probe.fs), dim(1));
            
            [X_image Z_image]=meshgrid(x_image, z_image);
            [X_org Z_org]=meshgrid(x_org, z_org);
            
            % --- for display
            obj.x_display=[x_start, x_end];
            obj.z_display=[z_start, z_end];
        end
        
        % ------------------------------------------------------------------
        function init_using_DA(obj)

            [obj.RF_aperture, obj.tstart, obj.tcompensation] = fct_zero_padding_RF_signals_DA(obj.RF_aperture);
            width = length(obj.RF_aperture);
            height = size(obj.RF_aperture{1}, 1);

            obj.RF_final = zeros([height, width]);
            
            for col=1:1:width
               obj.RF_final(:,col) = obj.RF_aperture{col}; 
            end
                        
        end
        
        % ------------------------------------------------------------------
        function [pres_in_vivio, x_disp, z_disp] = adapt_in_vivo(obj)
            
            % --- in vivo dimension
            x_img = obj.image.width * obj.image.CF;
            x_img = linspace(-x_img/2, x_img/2, obj.image.width);
            z_img = obj.image.height * obj.image.CF;
            z_img = linspace(0, z_img, obj.image.height);
                        
            % --- phantom dimension
            dim_phantom = size(obj.bmode);
            x_phantom = linspace(obj.x_display(1), obj.x_display(2), dim_phantom(2));
            z_phantom = linspace(obj.z_display(1), obj.z_display(2), dim_phantom(1));

            disp(['ADAPT_IN_VIVO ( z_phantom(1) = ' num2str(z_phantom(1)) ')']);
            disp(['ADAPT_IN_VIVO ( z_phantom(2) = ' num2str(z_phantom(2)) ')']);
            
            % --- interpolation
            [x_img  z_img] = meshgrid(x_img, z_img);
            [x_phantom z_phantom] = meshgrid(x_phantom, z_phantom);
            obj.in_vivo=interp2(x_img, z_img, double(obj.image.image), x_phantom, z_phantom, 'makima', 0);
            
            pres_in_vivio = fullfile(obj.path_res, [obj.param.phantom_name '_fit_in_vivo.png']);
            x_disp = obj.x_display;
            z_disp = obj.z_display;
            obj.in_vivo = fct_expand_histogram(obj.in_vivo, 0, 255);
%             obj.in_vivo = obj.in_vivo(:, 10:end-10);
%             obj.bmode = obj.bmode(:, 10:end-10);
            imwrite(uint8(obj.in_vivo), pres_in_vivio);
        
        end
        
        % ------------------------------------------------------------------
        function BF_CUDA_STA(obj, compounding)
            % Apply simple DAS/DMAS/DMAS+CF beamforming algorithm.
                          
            % --- analytic signal
            time_offset = fct_get_time_offset(obj.RF_aperture);
            % --- add zeros padding
            RF_signals = fct_zero_padding_RF_signals(obj.RF_aperture);
            % --- signal information
            [nb_sample, n_rcv] = size(RF_signals{1}); 
            dz = obj.probe.c/(2*obj.probe.fs);
            addpath(fullfile('..', 'mtl_synthetic_aperture'))
            % --- get image information
            [X_img, Z_img, obj.x_display, obj.z_display] = fct_get_grid_2D(obj.phantom, obj.image, obj.probe, [nb_sample, n_rcv], dz, obj.param);
            [n_points_z, n_points_x] = size(X_img);
            obj.low_res_image = zeros([n_points_z n_points_x obj.probe.Nelements]);
            % --- get probe position elements
            probe_width = (obj.probe.Nelements-1) * obj.probe.pitch;
            probe_pos_x = linspace(-probe_width/2, probe_width/2, obj.probe.Nelements);
            % --- number of active elements
            Nactive_tx = obj.param.Nactive;
            % --- apodization window
            apodization = fct_get_apodization([nb_sample, n_rcv], Nactive_tx, obj.probe.pitch, 'hanning_adaptative', 2, dz);
%             apodization = fct_interpolation(apodization, X_RF, Z_RF, X_img, Z_img);
%             apodization = ones(size(apodization));
            % --- time of flight matrix
            tof = fct_get_time_of_flight(X_img, Z_img, probe_pos_x, obj.probe.c);
            % --- define the CUDA module and kernel
            cuda_module_path_and_file_name = fullfile('..', 'cuda', 'bin', 'bfLowResRF.ptx');
            cuda_kernel_name = 'das_low_res';
            % --- get the CUDA kernel from the CUDA module
            cuda_kernel = parallel.gpu.CUDAKernel(cuda_module_path_and_file_name,...
            'double*, const double*, const int, const int, const int, const int, const double, const double, const double*, const int, const double*, const double',...
            cuda_kernel_name);
            % --- define grid and block size
            BLOCK_DIM_X = 32;
            BLOCK_DIM_Y = 32;
            cuda_kernel.GridSize = [round( ((n_points_x-1)/BLOCK_DIM_X) + 1), round( ((n_points_z-1)/BLOCK_DIM_Y) + 1), 1];
            cuda_kernel.ThreadBlockSize = [BLOCK_DIM_X, BLOCK_DIM_Y, 1];
            % --- initialization for kernel
            nb_rx        = int32(obj.probe.Nelements);
            nb_sample    = int32(nb_sample);
            imageW       = int32(n_points_x);
            imageH       = int32(n_points_z);
            c            = obj.param.c;
            fs           = double(obj.probe.fs);
            apodization  = double(apodization);
            tof          = double(tof);

            for id_tx=1:1:obj.probe.Nelements
                I = double(zeros([n_points_z n_points_x]));
                RF = RF_signals{id_tx};
%                 t_offset = double(time_offset(id_tx));
                t_offset = double(-time_offset(id_tx));
                % --- call the kernel
                I = feval(cuda_kernel, I, RF, nb_rx, nb_sample, imageW, imageH, c, fs, apodization, id_tx-1, tof, t_offset);
                % --- gather the output back from GPU to CPU
                I = gather(I);
%                 if id_tx == 35
                    figure(1)
                    imagesc(abs(I))
                    title(num2str(id_tx))
                    colorbar()
%                     a=1
%                 end
                % --- store low res image
                obj.low_res_image(:,:,id_tx) = I;
            end
            % --- compounding
            switch compounding
                case 'DAS'
                    obj.RF_final = sum(obj.low_res_image, 3);
                case 'DMAS'
%                     low_res_image_ = abs(obj.low_res_image);
                    low_res_image_ = obj.low_res_image; 
                    for id_col = 1:size(low_res_image_, 2)
                        % --- preserve dimensionality
                        cols_ = low_res_image_(:,id_col,:);
                        % --- compute yDMAS
                        col_1 = sum(cols_, 3);
                        col_1 = col_1.^2;
                        col_2 = cols_.^2;
                        col_2 = sum(col_2, 3);         
                        % --- store result
                        y_DMAS = col_1-col_2;
                        obj.RF_final(:, id_col) = y_DMAS;
                    end
            end
            
            figure()
            imagesc(20*log10(abs(hilbert(obj.RF_final)))+40)
        end
    end
    
end

% ------------------------------------------------------------------------------------------------------------------------------
function plot_low_res_images(m, n, low_res_img, path_res)
    nb_low_res=size(low_res_img, 3);
    div=nb_low_res/(m*n);
    assert(div>1,'Not enough images to plot.')
    div=floor(div);
    f=figure('visible', 'off');
    for id_tx=1:(m*n)
        
        img_=abs(low_res_img(:,:,id_tx*div));
        img_=img_.^(0.3);
        
        subplot(m,n,id_tx)
        imagesc(img_)
        set(gcf, 'Position', get(0, 'Screensize'));
        colormap gray
        title(['Low img emit., id_{tx}: ' num2str(id_tx*div)], 'FontSize', 6)
        
    end
    saveas(f, path_res);
    close(f)
end

% ------------------------------------------------------------------------------------------------------------------------------
function plot_apodized_window(m, n, apod_window, path_res, pos)

    nb_apod=size(apod_window, 3);
    div=nb_apod/(m*n);
    assert(div>1,'Not enough images to plot.')
    div=floor(div);
    f=figure('visible', 'off');
    rpath = fullfile(path_res, ['apod_' pos '.png']);
    for id_rx=1:(m*n)     
        subplot(m,n,id_rx)
        imagesc(apod_window(:,:,id_rx*div))
        colorbar
        set(gcf, 'Position', get(0, 'Screensize'));
        title(['apod window., id_{pos}: ' num2str(id_rx*div)],'fontsize',6)  
    end
    
    saveas(f, rpath);
end

% ------------------------------------------------------------------------------------------------------------------------------
function plot_low_res_images_unit(img, id, path_res)


    f=figure('visible', 'off');
    img_=abs(img);
%     img_=img_.^(0.3);
    img_= log(0.1+img); 
    imagesc(img_)
    colormap gray
    title(['Low img emit., id_tx: ' num2str(id)])
    
    saveas(f, fullfile(path_res, strcat('low_res_', num2str(id), '.png')));
    
end

% ------------------------------------------------------------------------------------------------------------------------------
