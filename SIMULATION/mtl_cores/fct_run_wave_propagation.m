function fct_run_wave_propagation(varargin)
    fclose all;

    switch nargin
        
      case 3
        path_param = varargin{1};
        path_phantom = varargin{2};
        id_tx = varargin{3};
        id_tx_end = varargin{3};

      case 4 
        path_param = varargin{1};
        path_phantom = varargin{2};
        id_tx = varargin{3};
        id_tx_end = varargin{4};

      otherwise
        error('Problem with parameters (fct_run_wave_propagation)')
    end

    % --- add path
    if ~isdeployed
        run(fullfile('..', 'mtl_utils', 'add_path.m'))
    end

    if isstring(id_tx)
        id_tx = str2double(id_tx);
    end

    if ischar(id_tx)
        id_tx = str2double(id_tx);
    end
    
    if isstring(id_tx_end)
        id_tx_end = str2double(id_tx_end);
    end
    
    if ischar(id_tx_end)
        id_tx_end = str2double(id_tx_end);
    end

    % --- create object
    simulation_obj=wavePropagation(path_param, path_phantom);
    name_phantom=split(path_phantom, '/');
    name_phantom=name_phantom{end};

    % --- launch simulation
    for i=id_tx:1:id_tx_end

      switch simulation_obj.param.soft
          case 'SIMUS'
              if simulation_obj.param.mode(1) %                                                     % run scanline-based acquisition
                  simulation_obj.scanline_based_simus(i);
              elseif simulation_obj.param.mode(2)                                                   % run scanline-based acquisition
                  simulation_obj.synthetic_aperture_simus(i);
              end

          case 'FIELD'
              if simulation_obj.param.mode(1)                                                       % run scanline based acquisition
                  simulation_obj.scanline_based_field(i);
              elseif simulation_obj.param.mode(2) && simulation_obj.param.dynamic_focusing == 0     % run synthetic-aperture acquisition
                  simulation_obj.synthetic_aperture_field(i);
              elseif simulation_obj.param.mode(2) && simulation_obj.param.dynamic_focusing == 1     % run synthetic-aperture acquisition
                  simulation_obj.dynamic_acquisition_field(i);
              end
      end

        % --- save RF data and probe
        if simulation_obj.param.save_exec_time == true
            simulation_obj.save_exec_time(name_phantom, i)
        end
        simulation_obj.save_raw_data(name_phantom, i);
    
        if i == 1
            simulation_obj.save_probe()
        end

    end
    
end
