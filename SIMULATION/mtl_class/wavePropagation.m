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
            obj.probe                 = fct_get_probe(obj.param.Nelements, obj.param.c/obj.param.fc);
            obj.probe.fnumber         = obj.param.fnumber;
            obj.probe.c               = obj.param.c;
            obj.probe.fc              = obj.param.fc;
            obj.probe.Nelements       = obj.param.Nelements;
            obj.probe.fs              = obj.param.fsCoef*obj.probe.fc;
            obj.sub_probe             = obj.probe;
            obj.sub_probe.Nelements   = obj.param.Nactive;
            obj.apod                  = hanning(obj.sub_probe.Nelements)';
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
                obj.RF_aperture=simus([obj.phantom.x_scatt; obj.phantom.x_max; obj.phantom.x_min],... % MUST function
                                      [obj.phantom.y_scatt; 0; 0],...
                                      [obj.phantom.z_scatt + obj.param.shift; obj.phantom.z_max; obj.phantom.z_min],...
                                      [obj.phantom.RC_scatt; 0; 0],...
                                      delay_probe,...
                                      obj.probe,...
                                      opt);  
                time_offset = zeros(size(obj.RF_aperture, 2), 1)';      % add 0 since there is not delay with simus
                obj.RF_aperture=[time_offset; obj.RF_aperture];
            obj.exec_time = toc(now1);
        end
        
        % ------------------------------------------------------------------
        function synthetic_aperture_field(obj, id_tx)
            
            field_init(-1)
            [emit_aperture, receive_aperture, time_compensation]=obj.init_field(50);
            % --- adapt phantom in order to fit field
            positions=[obj.phantom.x_scatt, obj.phantom.y_scatt, obj.phantom.z_scatt + obj.param.shift];
            now1 = tic();
            % --- display
            disp(['FIELD aperture ID (Synthetic aperture): ' num2str(id_tx)]);
            % --- apodization
            apo_tx = zeros(1, obj.probe.Nelements);
            apo_tx(id_tx) = 1;
            apo_rx = ones(1, obj.probe.Nelements);
            xdc_apodization(emit_aperture, 0, apo_tx);
            xdc_apodization(receive_aperture, 0, apo_rx);
            % --- we run the simulation
            [obj.RF_aperture, tstart] = calc_scat_multi(emit_aperture, receive_aperture, positions, obj.phantom.RC_scatt);  
            obj.exec_time = toc(now1);
            % --- add zero padding to RF signal
            time_offset = ones(size(obj.RF_aperture, 2), 1)' * (tstart - time_compensation);
            obj.RF_aperture=[time_offset; obj.RF_aperture];
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
            Tg = 1/obj.probe.fc;                % signal g(t) duration [sec]
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
            e=sin(2*pi*obj.probe.fc*te).*hanning(Ne)';      % excitation signal
            % --- set the excitation to the emission aperture
            xdc_excitation(emit_aperture, e);            
            time_compensation = conv(conv(e, g), g);
            time_compensation = length(time_compensation) / 2 / obj.probe.fs;
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
            dim = size(obj.RF_aperture, 1);
            obj.RF_aperture = reshape(obj.RF_aperture, [dim, 192, 192]);
            obj.RF_aperture = vertcat((tstart - time_compensation)*ones([1 192, 192]), obj.RF_aperture);
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
            % Save probe (it contains Fc, Fs...)
            path_probe=fullfile(obj.param.path_res, 'raw_data', [obj.param.phantom_name '_probe.mat']);
            path_sub_probe=fullfile(obj.param.path_res, 'raw_data', [obj.param.phantom_name '_subProbe.mat']);
            probe=obj.probe;
            sub_probe=obj.sub_probe;
            % --- save probe
            save(path_probe, 'probe');
            save(path_sub_probe, 'sub_probe');
        end
        
        % ------------------------------------------------------------------
    end
    
end