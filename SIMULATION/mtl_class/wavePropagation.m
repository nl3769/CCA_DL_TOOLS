classdef wavePropagation < handle 
   
    properties
        param;
    end
    
    properties (Access = private)
        exec_time;
        phantom;
        probe;
        apod;
        sub_probe;
        tx_delay;
        RF_aperture;
    end
    
    methods
        
        % ------------------------------------------------------------------
        function obj=wavePropagation(path_param, path_phantom)
            % --- constructor
            
            obj.phantom               = fct_load_phantom(path_phantom);
            obj.param                 = fct_load_param(path_param);
            obj.probe                 = getparam(obj.param.probe_name);
            obj.probe.fnumber         = obj.param.fnumber;
            obj.probe.c               = obj.param.c;
            obj.probe.fc              = obj.param.fc;
            obj.probe.Nelements       = obj.param.Nelements;
            obj.probe.fs              = obj.param.fsCoef*obj.probe.fc;
            obj.sub_probe             = obj.probe;
            obj.sub_probe.Nelements   = obj.param.Nactive;
            obj.tx_delay              = txdelay(0, obj.phantom.depth_of_focus, obj.sub_probe);
            obj.apod                  = hanning(obj.sub_probe.Nelements)';
        end
        
        % ------------------------------------------------------------------        
        function scanline_based_simus(obj, id_tx)
            % US simulation using MUST            
            
            % --- option for simus
            opt.WaitBar = false;
            opt.ParPool = false;
                        
            % --- scanline-based acquisition
            now1 = tic();
%             for id=1:(obj.probe.Nelements-obj.sub_probe.Nelements)
            disp(['SIMUS aperture ID (scanline-based): ' num2str(id_tx)]);
            % --- delay
            delay_probe=NaN(1, obj.probe.Nelements);
            delay_probe(id_tx:id_tx+obj.param.Nactive-1)=obj.tx_delay;
            % --- apodization
            apod_probe=zeros(1, obj.probe.Nelements);
            apod_probe(id_tx:id_tx+obj.param.Nactive-1)=obj.apod;
            obj.probe.TXapodization=apod_probe;
            % --- we run simus
%                del.delay_probe=delay_probe;
%                 fct_save_delay(['simus-' num2str(id)], delay_probe)
%                 fct_save_apo(['simus-' num2str(id)], apod_probe);
            obj.RF_aperture=simus([obj.phantom.x_scatt; obj.phantom.x_max; obj.phantom.x_min], ... % MUST function
                                  [obj.phantom.y_scatt; 0; 0], ...
                                  [obj.phantom.z_scatt; obj.phantom.z_max; obj.phantom.z_min], ...
                                  [obj.phantom.RC_scatt; 0; 0], ...
                                  delay_probe, ...
                                  obj.probe, ...
                                  opt);  

            time_offset = zeros(size(obj.RF_aperture, 2), 1)'; % since there is not delay using simus
            obj.RF_aperture=[time_offset; obj.RF_aperture];
            obj.exec_time = toc(now1);
        end
        
        % ------------------------------------------------------------------
        function synthetic_aperture_simus(obj, id_tx)
            % US simulation using MUST
            
            % --- option for simus
            opt.WaitBar = false;
            opt.ParPool = false;
                        
            % --- synthetic aperture acquisition
            now1 = tic();

            disp(['SIMUS aperture ID (synthetic aperture): ' num2str(id_tx)]);

            % --- delay
            delay_probe=NaN(1, obj.probe.Nelements);
            delay_probe(id_tx)=0;
            apod_probe=zeros(1, obj.probe.Nelements);
            apod_probe(id_tx)=1;
            obj.probe.TXapodization=apod_probe;
            % --- we run simus
%             if id_tx > floor(obj.param.Nactive/2) && id_tx < (obj.param.Nelements - floor(obj.param.Nactive/2) + 1)
                obj.RF_aperture=simus([obj.phantom.x_scatt; obj.phantom.x_max; obj.phantom.x_min],... % MUST function
                                      [obj.phantom.y_scatt; 0; 0],...
                                      [obj.phantom.z_scatt; obj.phantom.z_max; obj.phantom.z_min],...
                                      [obj.phantom.RC_scatt; 0; 0],...
                                      delay_probe,...
                                      obj.probe,...
                                      opt);  
                time_offset = zeros(size(obj.RF_aperture, 2), 1)';      % since there is not delay with simus
                obj.RF_aperture=[time_offset; obj.RF_aperture];
%             else
%                 obj.RF_aperture = zeros(obj.param.Nelements, obj.param.Nelements);
%             end
            
            obj.exec_time = toc(now1);

        end
        
        % ------------------------------------------------------------------
        function scanline_based_field(obj, id_tx)
            
            field_init(-1)
            [emit_aperture, receive_aperture, time_compensation]=obj.init_field(obj.phantom.depth_of_focus);
           
            % --- adapt phantom in order to fit field
            positions=[obj.phantom.x_scatt, obj.phantom.y_scatt, obj.phantom.z_scatt];
            
            % --- probe dimension
            image_width=(obj.probe.Nelements-1)*(obj.probe.pitch);
            dx = obj.probe.pitch; 

%             image_width/(obj.probe.Nelements-1);
            xstart=-image_width/2;
            % --- scanline-based acquisition
            now1 = tic();
            % --- get xcurrent
            xcurrent = xstart + (id_tx-1 + (obj.sub_probe.Nelements-1)/2 - 1 ) * dx;
            % --- display
            disp(['FIELD aperture ID (scanline-based): ' num2str(id_tx)]);
            % --- focuse depth
            focus_point_emit = [xcurrent 0 obj.phantom.depth_of_focus];
            % --- set the origin for emission and reception
            xdc_center_focus(emit_aperture, [xcurrent 0 0]);         
            % --- create the focus time line for emission and reception   
            xdc_focus(emit_aperture, 0, focus_point_emit);
            % --- apodization window for emission and reception
            apo = [zeros(1, id_tx-1) hanning(obj.sub_probe.Nelements)' zeros(1, obj.probe.Nelements-obj.sub_probe.Nelements-id_tx+1)];
            xdc_apodization(emit_aperture, 0, apo);
            % --- get delay
            delay = xdc_get(emit_aperture, 'focus');
            %figure(1)
            %plot(delay)
            delay = delay(id_tx:id_tx+obj.sub_probe.Nelements-1);
            %disp(['min(delay): ' num2str(min(delay))]);
%             a= -1.1253e-06;
%             figure(1)
%             plot(delay)
            % --- we run the simulation
            [obj.RF_aperture, tstart] = calc_scat_multi(emit_aperture, receive_aperture, positions, obj.phantom.RC_scatt);  
            % --- add zero padding to RF signal
%             obj.RF_aperture = [zeros(size(obj.RF_aperture, 2), round((tstart - time_compensation - min(delay)) * obj.sub_probe.fs))'; obj.RF_aperture];
            obj.RF_aperture = [zeros(size(obj.RF_aperture, 2), round((tstart - time_compensation - min(delay)) * obj.sub_probe.fs))'; obj.RF_aperture];
            
            % --- get info
%                 apo_=xdc_get(emit_aperture, 'apo');
%                 delay=xdc_get(emit_aperture, 'focus');  
%                 fct_save_delay(['field-' num2str(id)], delay(2:end, 1))
%                 fct_save_apo(['field-'   num2str(id)], apo)
%             end
            % --- simulation time

            obj.exec_time = toc(now1);
            
        end
        
        % ------------------------------------------------------------------
        function synthetic_aperture_field(obj, id_tx)
            
            field_init(-1)
            [emit_aperture, receive_aperture, time_compensation]=obj.init_field(50);
            
%             % --- adapt tx id
%             id_tx_active = id_tx + floor(obj.param.Nactive/2);

            % --- adapt phantom in order to fit field
            positions=[obj.phantom.x_scatt, obj.phantom.y_scatt, obj.phantom.z_scatt];
            
            now1 = tic();

            % --- display
            disp(['FIELD aperture ID (Synthetic aperture): ' num2str(id_tx)]);
            
            % --- apodization
            apo_tx = zeros(1, obj.probe.Nelements);
            apo_tx(id_tx) = 1;
            apo_rx = ones(1, obj.probe.Nelements);
            xdc_apodization(emit_aperture, 0, apo_tx);
            xdc_apodization(receive_aperture, 0, apo_rx);
            
%             if id_tx > floor(obj.param.Nactive/2) && id_tx < (obj.param.Nelements - floor(obj.param.Nactive/2) + 1)
                % --- we run the simulation
                [obj.RF_aperture, tstart] = calc_scat_multi(emit_aperture, receive_aperture, positions, obj.phantom.RC_scatt);  
                obj.exec_time = toc(now1);

                % --- add zero padding to RF signal
                time_offset = ones(size(obj.RF_aperture, 2), 1)' * (tstart - time_compensation);
                obj.RF_aperture=[time_offset; obj.RF_aperture];
%             else
%                 obj.RF_aperture = zeros(obj.param.Nelements, obj.param.Nelements);
%             end
        end
        
        % ------------------------------------------------------------------
        function [emit_aperture, receive_aperture, time_compensation]=init_field(obj, depth_of_focus)
            
            % --- set the sampling frequency
            set_sampling(obj.probe.fs);
            Ts=1/obj.probe.fs;
            % --- set sound speed
            set_field('c', obj.param.c);
            % -- emission and reception transducer
            focus_emit=[0 0 depth_of_focus];
            
            emit_aperture=xdc_linear_array(obj.probe.Nelements, obj.probe.width, obj.probe.height, obj.probe.kerf, 15, 15, focus_emit);
            receive_aperture=xdc_linear_array(obj.probe.Nelements, obj.probe.width, obj.probe.height, obj.probe.kerf, 15, 15, focus_emit); % focus_emit is removed below
            xdc_focus_times(receive_aperture, 0, zeros(1, obj.probe.Nelements));
            xdc_focus_times(emit_aperture, 0, zeros(1, obj.probe.Nelements));
            % --- transducer impulse response g(t)
            Bg = 10e6;                          % transducer spectral bandwidth [Hz] -> unused
            Tg = 2/obj.probe.fc;                % signal g(t) duration [sec]
            tg = 0:Ts:Tg;                       % sampled time vector [sec]
            if(~rem(length(tg), 2)), tg=[tg tg(end)+Ts]; end        % even Ng
            Ng = length(tg);                      % sampled time vector length [number of samples]
            g = sin(2*pi*obj.probe.fc*tg).*hanning(Ng)';
            % --- Set the aperture transducer impulse response
            xdc_impulse(emit_aperture, g);
            xdc_impulse(receive_aperture, g);
            % --- excitation signal e(t) to the emission aperture
            Te=1/obj.probe.fc;                              % signal duration [sec]
            te=0:Ts:Te;                                     % sampled time vector [sec]
            Ne=length(te);                                  % sampled time vector length [number of samples]
%             e=sin(2*pi*obj.probe.fc*te).*hanning(Ne)';      % excitation signal
            e=sin(2*pi*obj.probe.fc*te);      % excitation signal
            % --- set the excitation to the emission aperture
            xdc_excitation(emit_aperture, e);
            
            time_compensation = conv(conv(e, g), g);
            time_compensation = length(time_compensation) / 2 / obj.probe.fs;

            % --- COMPUTE CONVOLUTION IN FOURIER DOMAIN
%             e_f=fft(e);

%             t_t = conv2(e,g);
%             t_f = fftshift(fft(t_t));
%             n = length(t_t);
%             f = (-n/2:n/2-1)*(obj.probe.fs/n);

%             plot(f(round(length(f)/2):end),abs(t_f(round(length(f)/2):end)))
%             xlabel('Frequency (Hz)')
%             ylabel('Magnitude');
%             fe_shift = (-Ne/2:Ne/2-1)*(obj.probe.fs/Ne);
%             e_f_shift = fftshift(fft(e));
%             plot(fe_shift,abs(e_f_shift))
%             xlabel('Frequency (Hz)')
%             ylabel('Magnitude');
        
%         n = length(x);                         
%         fshift = (-n/2:n/2-1)*(fs/n);
%         yshift = fftshift(y);
%         plot(fshift(round(length(fshift)/2):end),abs(yshift(round(length(fshift)/2):end)))
%         xlabel('Frequency (Hz)')
%         ylabel('Magnitude')
        
        end
        
        % ------------------------------------------------------------------
        function calc_scatt_all_field(obj)
                        
            field_init(-1)
            set_field('show_times', 30)
            [emit_aperture, receive_aperture, time_compensation]=obj.init_field(10);
            % --- set scatterers position
            positions=[obj.phantom.x_scatt, obj.phantom.y_scatt, obj.phantom.z_scatt];
            % --- apodization window for emission and reception
            if ~isdeployed
                addpath(fullfile('..', 'mtl_synthetic_aperture'))
            end

            tx_apod = ones([1 192]);
            rx_apod = ones([1 192]);
            xdc_apodization(emit_aperture, 0, tx_apod);
            xdc_apodization(receive_aperture, 0, rx_apod);
            xdc_times_focus(receive_aperture, 0, zeros([1 192]));
            xdc_times_focus(emit_aperture, 0, zeros([1 192]));
            % --- we run the simulation
            now1 = tic();
            [obj.RF_aperture, tstart] = calc_scat_all(emit_aperture, receive_aperture, positions, obj.phantom.RC_scatt, 1);  
            % --- add zero padding to RF signal 
%             obj.RF_aperture=[tstart; time_compensation; obj.RF_aperture];
            dim = size(obj.RF_aperture, 1);
            obj.RF_aperture = reshape(obj.RF_aperture, [dim, 192, 192]);
            
%             obj.RF_aperture = vertcat(zeros([1 192, 192]), obj.RF_aperture);
            obj.RF_aperture = vertcat((tstart - time_compensation)*ones([1 192, 192]), obj.RF_aperture);
            
            % --- simulation time
            obj.exec_time = toc(now1);        
            
        end
        
        % ------------------------------------------------------------------
        function dynamic_focus_field(obj, id_tx)
                        
            field_init(-1)
            set_field('show_times', 30)
            [emit_aperture, receive_aperture, time_compensation]=obj.init_field(10);
            % --- adapt id_tx
%             id_tx_active = id_tx + floor(obj.param.Nactive/2);
            id_tx_active = id_tx;
%             id_tx_active = id_tx;
            % --- set scatterers position
            positions=[obj.phantom.x_scatt, obj.phantom.y_scatt, obj.phantom.z_scatt];
            % --- probe dimension
            probe_width = (obj.probe.Nelements-1) * (obj.probe.pitch);
            dx = obj.probe.pitch;
            xstart = -probe_width/2;
            % --- compute xcurrent
%             xcurrent = xstart + (id_tx-1+(obj.sub_probe.Nelements-1)/2) * dx;
            xcurrent = xstart + (id_tx-1) * dx;
            % --- display
            disp(['FIELD aperture ID (Dynamic Focusing): ' num2str(id_tx_active)]);
            % --- set the origin for emission and reception
            xdc_center_focus(emit_aperture, [xcurrent 0 0]);    
            xdc_center_focus(receive_aperture, [xcurrent 0 0]);
            % --- apodization window for emission and reception
            if ~isdeployed
                addpath(fullfile('..', 'mtl_synthetic_aperture'))
            end
            % --- set apodization
            dz = obj.probe.c/(2*obj.probe.fs);
            max_dist=round((sqrt(max(obj.phantom.z_max)^2 + (obj.phantom.x_max - obj.phantom.x_min)^2))/dz);
            dim=[max_dist, obj.probe.Nelements];
            apodization = fct_get_apodization(dim, obj.sub_probe.Nelements, obj.sub_probe.pitch, 'hanning_adaptative', obj.param.fnumber, dz);
            tx_apod = squeeze(apodization(:,:,id_tx_active));
            rx_apod = squeeze(apodization(:,id_tx_active,:));
%             rx_apod = zeros(size(tx_apod));
%             for id_rx=1:size(rx_apod, 2)
%                 rx_apod(:,id_rx) = apodization(:, id_rx, id_tx_active);
%             end
            tx_apod = ones(size(tx_apod ));
            rx_apod = ones(size(rx_apod ));
            tsamples = (1:1:max_dist)'*1/(obj.param.fc*obj.param.fsCoef);
            xdc_apodization(emit_aperture, tsamples, tx_apod);
            xdc_apodization(receive_aperture, tsamples, rx_apod);
            % --- create the dynamic focus time line for emission and reception   
            xdc_dynamic_focus(emit_aperture, 0, 0, 0);
            xdc_dynamic_focus(receive_aperture, 0, 0, 0);
            % --- we run the simulation
            now1 = tic();
            [obj.RF_aperture, tstart] = calc_scat(emit_aperture, receive_aperture, positions, obj.phantom.RC_scatt);  
            % --- add zero padding to RF signal 
            obj.RF_aperture=[tstart; time_compensation; obj.RF_aperture];
            % --- simulation time
            obj.exec_time = toc(now1);        
            
        end
        
        % ------------------------------------------------------------------
        function save_raw_data(obj, name_phantom, id_tx)
            
            raw_data=obj.RF_aperture;
            name_=split(name_phantom, '.');
            name_=name_{1};
            path_raw_data=fullfile(obj.param.path_res, 'raw_data', 'raw_', [name_ '_raw_data_id_tx_' num2str(id_tx) '.mat']);
            save(path_raw_data, 'raw_data', '-v7.3'); % flag '-v7.3' to store data larger than 2Go

        end
        
        % ------------------------------------------------------------------
        function save_exec_time(obj, name_phantom, id_tx)
            % Save execution time. 
            time_disp=['Execution time (wavePropagation):' num2str(obj.exec_time/60/60) ' hours.'];
            disp(time_disp);
            name_=split(name_phantom, '.');
            name_=name_{1};
            fct_save_string(fullfile(obj.param.path_res, 'raw_data', 'exec_'), ['exec_time_' name_ '_id_tx_' num2str(id_tx)], time_disp);
        end
        
        % ------------------------------------------------------------------
        function save_probe(obj)
            
            path_probe=fullfile(obj.param.path_res, 'raw_data', [obj.param.phantom_name '_probe.mat']);
            path_sub_probe=fullfile(obj.param.path_res, 'raw_data', [obj.param.phantom_name '_subProbe.mat']);
            probe=obj.probe;
            sub_probe=obj.sub_probe;
            
            save(path_probe, 'probe');
            save(path_sub_probe, 'sub_probe');
        end
        
        % ------------------------------------------------------------------
    end
    
end

% --- get apodization window
function [apod_window] = hanning_adaptative(dim, Nactive, pitch, f_number, dz, id_tx)

    width = dim(2);
    height = dim(1);

    apod_window_ = zeros([height, Nactive]);
    apod_window = zeros([height, width]);
    x_probe = -(Nactive-1)*pitch/2:pitch:(Nactive-1)*pitch/2;
    
    % --- Create apodization window of size depth*Nelements
    for id_z=1:1:height
        z = id_z*dz;
        Nactive_ = round(z/(pitch*f_number));

        if Nactive_ < 3 
            Nactive_ = 3 ; 
        elseif Nactive_ >= Nactive 
            Nactive_ = Nactive;
        end

        apod_act_int = hanning(Nactive_*10)';
        x_act_int = -(Nactive_*10-1)*pitch/10/2 : pitch/10 : (Nactive_*10-1)*pitch/10/2;

        x_int = -(Nactive * 10 - 1) * pitch/10/2 : pitch/10 : (Nactive * 10 - 1) * pitch/10/2;
        apod_int = interp1(x_act_int, apod_act_int, x_int, 'linear', 0);

        apod = interp1(x_int, apod_int, x_probe);
        
        % --- sum has to equal equal to zero
        sum_ = sum(apod);
        
        apod_window_(id_z, :) = apod/sum_;

    end          

%     % --- window offset according to the element
%     if id_tx >= Nactive/2 && id_tx < (width - Nactive/2 +1)
%         id = floor(id_tx - Nactive/2 +1);
%         apod_window(:, id:id+Nactive-1) = apod_window_;
%     end
    
end
