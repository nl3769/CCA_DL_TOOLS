function fct_run_wave_propagation(varargin)
    fclose all;
    
    % we select how many elements emit
    switch nargin
      case 4
        path_param = varargin{1};
        path_phantom = varargin{2};
        flag_cluster = varargin{3};
        id_tx = varargin{4};
        id_tx_end = varargin{4};

      case 5 
        path_param = varargin{1};
        path_phantom = varargin{2};
        flag_cluster = varargin{3};
        id_tx = varargin{4};
        id_tx_end = varargin{5};

      otherwise
        error('Problem with parameters (fct_run_wave_propagation)')
    end

    % --- add path
    if ~isdeployed
        run('../package_utils/add_path.m')
    end

    if isstring(id_tx)
        id_tx = str2double(id_tx);
    end
    
    if isstring(id_tx_end)
        id_tx_end = str2double(id_tx_end);
    end
    
    % --- create object
    simulation_obj=wavePropagation(path_param, path_phantom, flag_cluster);
    name_phantom=split(path_phantom, '/');
    name_phantom=name_phantom{end};

    % --- launch simulation
    for i=id_tx:1:id_tx_end

      switch simulation_obj.param.soft
          case 'SIMUS'
              if simulation_obj.param.mode(1) % run scane-line based acquisition
                  simulation_obj.scanline_based_simus(i);
              elseif simulation_obj.param.mode(2)
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
        simulation_obj.save_exec_time(name_phantom, i)
        simulation_obj.save_raw_data(name_phantom, i);
    
        if i == 1
            simulation_obj.save_probe()
        end

    end
    
end
