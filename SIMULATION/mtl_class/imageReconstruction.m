classdef imageReconstruction < handle 
   
    properties
        param;
    end
    
    properties (Access=private)
        RF_signals;
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
        x_bmode;
        z_bmode;
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj=imageReconstruction(varargin)
            % Constructor
            
            switch nargin
                case 6
                    % varargin{1} -> (str) path_data
                    % varargin{2} -> (str) param_name
                    % varargin{3} -> (str) probe name
                    % varargin{4} -> (str) subprobe name
                    % varargin{5} -> (str) phantom_name
                    % varargin{6} -> (array of double) raw_data
                    [obj.probe, obj.sub_probe, obj.param, obj.phantom]=fct_get_data(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5});
                    obj.RF_signals=varargin{6};
                    obj.tstart=zeros(size(obj.RF_signals, 1), 65);
                    obj.tcompensation=0;
                    path_image_information=fullfile(obj.param.path_res, 'phantom', 'image_information.mat');
                    image=load(path_image_information);
                    obj.image=image.image;
                    obj.path_res=fct_create_directory(varargin{1}, 'results');
                otherwise
                    disp('Problem with parameters (imageReconstruction constructor)');
            end
            
        end
                
        % ------------------------------------------------------------------
        function get_bmode_gamma_correction(obj)
            % Get bmode image withtout applying log compression. Only gamma correction is applyed. 
            % --- RF to IQ
            if isfield(obj.param, "input_bf") && obj.param.input_bf == "IQ"
                obj.IQ=obj.RF_final ;
            else
                obj.IQ=hilbert(obj.RF_final);
            end            
            % --- apply TGC
            if obj.param.TGC==true
                obj.IQ=tgc(obj.IQ);
            end
            % --- compute envelope
            obj.bmode=abs(obj.IQ); % real envelope
            % --- apply gamma correction
            obj.bmode=obj.bmode - min(obj.bmode(:)); %/max(obj.bmode(:));
            obj.bmode=obj.bmode / max(obj.bmode(:));
            obj.bmode=obj.bmode.^(obj.param.gamma);
            % --- apply image adjustement
            obj.bmode=obj.bmode/max(obj.bmode(:));
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
            rpath=fullfile(obj.path_res, [obj.param.phantom_name '_in_silico_dimension.png']);
            % --- compute the dimension of the image
            dim=size(obj.bmode);
            x_disp=linspace(obj.x_display(1), obj.x_display(2), dim(2));
            z_disp=linspace(obj.z_display(1), obj.z_display(2), dim(1));
            % --- save the image
            f=figure('visible', 'off');
            imagesc(x_disp*1e3, z_disp*1e3, obj.bmode); 
            colormap gray; colorbar; axis image;
            title('bmode image'); xlabel('width in mm'); ylabel('height in mm')
            saveas(f, rpath);
            % --- path to save the in vivo signal
            rpath=fullfile(obj.path_res, [obj.param.phantom_name '_in_vivo_dimension.png']);                                                       
            % --- save the image
            f=figure('visible', 'off');
            imagesc(x_disp*1e3, z_disp*1e3, obj.in_vivo); 
            colormap gray; colorbar; axis image
            title('bmode image'); xlabel('width in mm'); ylabel('height in mm')
            saveas(f, rpath);
            % --- save 5F signal
            RF=obj.RF_final;
            rpath=fullfile(obj.path_res, [obj.param.phantom_name '_RF_data.mat']);  
            save(rpath, 'RF');
        end
        
        % ------------------------------------------------------------------
        function [psave, pres_sim]=save_bmode(obj, scatt)
            % Save bmode image image with the dimension of the phantom

            % --- we save the image
            pres_=fullfile(obj.path_res, [obj.param.phantom_name '_bmode_result_physical_dimension.png']);
            obj.bmode=fct_expand_histogram(obj.bmode, 0, 255);
            % --- compute the dimension of the image
            dim=size(obj.bmode);
            x_disp=linspace(obj.x_display(1), obj.x_display(2), dim(2));
            z_disp=linspace(obj.z_display(1), obj.z_display(2), dim(1));
            % --- save the image
            debug=false;
            if debug 
                f=figure();
                imagesc(x_disp*1e3, z_disp*1e3, obj.bmode); 
                hold on
                if scatt
                    plot(obj.phantom.x_scatt*1e3, obj.phantom.z_scatt*1e3, 'go','MarkerSize',10)
                end
                line([obj.x_display(1)*1e3,obj.x_display(1)*1e3*0.95],[DF*1e3, DF*1e3],'Color','r','LineWidth',1)
                line([obj.x_display(2)*1e3*0.95,obj.x_display(2)*1e3],[DF*1e3, DF*1e3],'Color','r','LineWidth',1)
                hold off; axis image; colormap gray;
                title('Bmode image'); xlabel('width in mm'); ylabel('height in mm');
                saveas(f, pres_);
            end
            % --- we save the image
            pres_sim=fullfile(obj.path_res, [obj.param.phantom_name '_bmode.png']);
            imwrite(uint8(obj.bmode), pres_sim);
            psave=obj.path_res;
        end
        
        % ------------------------------------------------------------------
        function init_using_DA(obj)

            [obj.RF_signals, obj.tstart, obj.tcompensation]=fct_zero_padding_RF_signals_DA(obj.RF_signals);
            width=length(obj.RF_signals);
            height=size(obj.RF_signals{1}, 1);
            obj.RF_final=zeros([height, width]);
            for col=1:1:width
               obj.RF_final(:,col)=obj.RF_signals{col}; 
            end
        end
        
        % ------------------------------------------------------------------
        function adapt_dimension(obj)
        
            % --- in vivo dimension
            x_img=obj.image.width * obj.image.CF;
            x_img=linspace(-x_img/2, x_img/2, obj.image.width);
            z_img=obj.image.height * obj.image.CF;
            z_img=linspace(0, z_img, obj.image.height);        
            
            % --- interpolation
            [x_img, z_img]=meshgrid(x_img, z_img);
            obj.in_vivo=interp2(x_img, z_img, double(obj.image.image), obj.x_bmode, obj.z_bmode - obj.param.shift, 'makima', 0);
            
        end
        
        % ------------------------------------------------------------------
        function [pres_in_vivio, x_disp, z_disp]=save_in_vivo(obj)
            
            pres_in_vivio=fullfile(obj.path_res, [obj.param.phantom_name '_fit_in_vivo.png']);
            x_disp=obj.x_display;
            z_disp=obj.z_display;
            in_vivo=fct_expand_histogram(obj.in_vivo, 0, 255);
            imwrite(uint8(in_vivo), pres_in_vivio);
        end
        
        % ------------------------------------------------------------------
        function BF_CUDA_STA(obj, compounding)
            % Apply simple DAS/DMAS/DMAS+CF beamforming algorithm.
                          
            % --- analytic signal
            time_offset=fct_get_time_offset(obj.RF_signals);
            % --- add zeros padding
            RF_signals=fct_zero_padding_RF_signals(obj.RF_signals);
            % --- signal information
            [nb_sample, n_rcv, ~]=size(RF_signals); 
            dz=obj.probe.c/(2*obj.probe.fs);
            addpath(fullfile('..', 'mtl_synthetic_aperture'))
            % --- get image information
            [X_img_bf, Z_img_bf, X_RF, Z_RF, obj.x_display, obj.z_display, n_pts_x, n_pts_z]=fct_get_grid_2D(obj.phantom, obj.image, obj.probe, [nb_sample, n_rcv], dz, obj.param);
%             [X_img_bf, Z_img_bf, X_RF, Z_RF, obj.x_display, obj.z_display, n_pts_x, n_pts_z]=fct_get_grid_2D_elisabeth(obj.phantom, obj.image, obj.probe, [nb_sample, n_rcv], dz, obj.param);
            [n_points_z, n_points_x]=size(X_img_bf);
            obj.low_res_image=zeros([n_points_z n_points_x obj.probe.Nelements]);
            % --- get probe position elements
            probe_width=(obj.probe.Nelements-1) * obj.probe.pitch;
            probe_pos_x=linspace(-probe_width/2, probe_width/2, obj.probe.Nelements);
            probe_pos_z=zeros(1, obj.probe.Nelements);
            % --- apodization window
            set_apod=false;
            if set_apod
                depth_focus=5;
                apodization=fct_get_apodization([nb_sample, n_rcv], obj.param.Nactive, obj.probe.pitch, 'hanning_adaptative', depth_focus, dz, t_offset);
                apodization=fct_interpolation(apodization, X_RF, Z_RF, X_img_bf, Z_img_bf);
            else
                apodization=ones( [n_points_z, n_points_x obj.probe.Nelements]);
            end
            % --- define the CUDA module and kernel
            obj.param.input_bf="IQ";
            if isfield(obj.param, "input_bf") && obj.param.input_bf == "IQ" 
                cuda_module_path_and_file_name=fullfile('..', 'cuda', 'bin', 'bfFullLowResImgIQ.ptx');
                cuda_kernel_name='bf_low_res_images';
                cuda_kernel=parallel.gpu.CUDAKernel(cuda_module_path_and_file_name,...
                'const double*, const double*, const double*, const double*, const int, const int, const int, const int, const double, const double, const double*, const double*, const double*, const double*, double*, double*',...
                cuda_kernel_name);
                % --- get IQ signal
                IQ_signals=fct_get_analytic_signals(RF_signals, obj.probe, time_offset);
                Iiq=imag(IQ_signals);
                Riq=real(IQ_signals);
                IlowRes=zeros(size(obj.low_res_image));
                RlowRes=zeros(size(obj.low_res_image));
            else
                cuda_module_path_and_file_name=fullfile('..', 'cuda', 'bin', 'bfFullLowResImgRF.ptx');
                cuda_kernel_name='bf_low_res_images';
                cuda_kernel=parallel.gpu.CUDAKernel(cuda_module_path_and_file_name,...
                'double*, double*, double*, double*, int, int, int, int, double, double, double*, double*, double*, double*',...
                cuda_kernel_name);
            end
            % --- define grid and block size
            BLOCK_DIM_X=8;
            BLOCK_DIM_Y=16;
            BLOCK_DIM_Z=8;
            cuda_kernel.GridSize=[round( ((n_points_x-1)/BLOCK_DIM_X) + 1), round( ((n_points_z-1)/BLOCK_DIM_Y) + 1), round( ((n_points_x-1)/BLOCK_DIM_Z) + 1)];
            cuda_kernel.ThreadBlockSize=[BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z];
            % --- initialization for kernel
            pos_x_img=X_img_bf(1,:);
            pos_z_img=Z_img_bf(:,1);
            nb_rx=int32(obj.probe.Nelements);
            nb_sample=int32(nb_sample);
            imageW=int32(n_points_x);
            imageH=int32(n_points_z);
            c=obj.param.c;
            fs=double(obj.probe.fs);
            apodization=double(apodization);
            % --- call kernel            
            if isfield(obj.param, "input_bf") && obj.param.input_bf == "IQ"
                [IlowRes, RlowRes]=feval(cuda_kernel, Iiq, Riq, probe_pos_x, probe_pos_z, nb_rx, nb_sample, imageW, imageH, c, fs, apodization, pos_x_img, pos_z_img, -time_offset, IlowRes, RlowRes);
                % --- gather the output back from GPU to CPU
                IlowRes=gather(IlowRes);
                RlowRes=gather(RlowRes);
                obj.low_res_image=RlowRes + IlowRes*1i;
            else
                obj.low_res_image=feval(cuda_kernel, obj.low_res_image, RF_signals, probe_pos_x, probe_pos_z, nb_rx, nb_sample, imageW, imageH, c, fs, apodization, pos_x_img, pos_z_img, -time_offset);
                % --- gather the output back from GPU to CPU
                obj.low_res_image=gather(obj.low_res_image);
            end            
            % --- compounding
            obj.compounding(compounding);            
            % --- real image domain
            x_img=linspace(X_img_bf(1,1), X_img_bf(1,end), n_pts_x);
            z_img=linspace(Z_img_bf(1,1), Z_img_bf(end,1), n_pts_z);
            [obj.x_bmode, obj.z_bmode]=meshgrid(x_img, z_img);
        end
        
        % -----------------------------------------------------------------
        function BF_DG(obj)
            % Apply simple DAS/DMAS/DMAS+CF beamforming algorithm.
                          
            % --- analytic signal
            time_offset=fct_get_time_offset(obj.RF_signals);
            % --- add zeros padding
            RF_signals=fct_zero_padding_RF_signals(obj.RF_signals);
            % --- signal information
            [nb_sample, n_rcv, ~]=size(RF_signals); 
            dz=obj.probe.c/(2*obj.probe.fs);
            addpath(fullfile('..', 'mtl_synthetic_aperture'))
            % --- get image information
            [X_img, Z_img, X_RF, Z_RF, obj.x_display, obj.z_display]=fct_get_grid_2D(obj.phantom, obj.image, obj.probe, [nb_sample, n_rcv], dz, obj.param);
            [n_points_z, n_points_x]=size(X_img);
            obj.low_res_image=zeros([n_points_z n_points_x obj.probe.Nelements]);
            % --- get probe position elements
            probe_width=(obj.probe.Nelements-1) * obj.probe.pitch;
            probe_pos_x=linspace(-probe_width/2, probe_width/2, obj.probe.Nelements);
            probe_pos_z=zeros(1, obj.probe.Nelements);
            % --- number of active elements
            Nactive_tx=obj.param.Nactive;
            % --- apodization window
            apodization=fct_get_apodization([nb_sample, n_rcv], Nactive_tx, obj.probe.pitch, 'hanning_adaptative', 5, dz);
            apodization=fct_interpolation(apodization, X_RF, Z_RF, X_img, Z_img);
            disp('Beamformation in progress...');
            probe_width=(obj.probe.Nelements-1) * obj.probe.pitch;
            arrayx=linspace(-probe_width/2, probe_width/2, obj.probe.Nelements);
            arrayz=zeros(1,length(arrayx));
            arrayy=zeros(1,length(arrayx));
            rxAptPos=[arrayx' arrayy' arrayz'];
            txAptPos=rxAptPos;
            param=struct();
            param.fs=obj.probe.fs;
            param.c=obj.probe.c;
            param.pitch=obj.probe.pitch;
            param.Pitch_x=obj.probe.pitch;
            param.Pitch_y=obj.probe.pitch;
            param.xm=rxAptPos(:,1);
            param.ym=rxAptPos(:,2);
            param.zm=rxAptPos(:,3);
            param.centers=rxAptPos;
            param.no_elements=obj.probe.Nelements;
  
            Y_img=zeros(size(X_img));
            bf_image=zeros(size(X_img)); 
            t_compensation=0;
            for aa=1:obj.probe.Nelements
                disp(['Beamforming low resolution image ',num2str(aa)]);
                param.xn= param.centers(aa,1);
                param.yn= param.centers(aa,2);
                param.zn= param.centers(aa,3);
                param.t0=time_offset(aa)-t_compensation;
                param.W=ones(1,obj.probe.Nelements); % Rect apodization (receive)
                [lowres]=rfbf_3D(RF_signals(:,:,aa), X_img, Y_img, Z_img, param);
                bf_image=bf_image+lowres;
            end
            env=abs(hilbert(bf_image));
            env=env/max(env(:));
            figure; imagesc(20*log10(env),[-50 0]); colormap gray
            obj.RF_final=bf_image;
            figure()
            imagesc(20*log10(abs(hilbert(obj.RF_final)))+40)
        end
        
        % -----------------------------------------------------------------
        function postprocessing(obj)
            
            % --- perform a low pass filtering
            obj.bmode=imgaussfilt(obj.bmode, 1, 'FilterSize', 3, 'Padding', 'symmetric');
            % --- perform a median filtering
            obj.bmode=medfilt2(obj.bmode);
            % --- perform a low pass filtering
            obj.bmode=imgaussfilt(obj.bmode, 1, 'FilterSize', 3, 'Padding', 'symmetric');    
            % --- apply treshold
            obj.bmode=imadjust(obj.bmode, [obj.param.imadjust_vals(1) obj.param.imadjust_vals(2)], []);
        end
        
        % -----------------------------------------------------------------
        function compounding(obj, mode)
            
            switch mode
                case 'DAS'
                    obj.RF_final=sum(obj.low_res_image, 3);
                case 'DMAS'
                    low_res_image_=obj.low_res_image; 
                    for id_col=1:size(low_res_image_, 2)
                        % --- preserve dimensionality
                        cols_=low_res_image_(:,id_col,:);
                        % --- compute yDMAS
                        col_1=sum(cols_, 3);
                        col_1=col_1.^2;
                        col_2=cols_.^2;
                        col_2=sum(col_2, 3);         
                        % --- store result
                        y_DMAS=col_1-col_2;
                        obj.RF_final(:, id_col)=y_DMAS;
                    end
            end
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
    rpath=fullfile(path_res, ['apod_' pos '.png']);
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
    img_= log(0.1+img); 
    imagesc(img_)
    colormap gray
    title(['Low img emit., id_tx: ' num2str(id)])
    saveas(f, fullfile(path_res, strcat('low_res_', num2str(id), '.png')));
end
% ------------------------------------------------------------------------------------------------------------------------------
