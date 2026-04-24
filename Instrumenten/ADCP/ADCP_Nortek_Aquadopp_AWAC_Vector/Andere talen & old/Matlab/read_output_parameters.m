function out = read_output_parameters(dpl, out_param)

    % out = READ_OUTPUT_PARAMETERS(dpl, output_param) reads output
    % parameters defined in out_param, checks if values are corrects,
    % assigns default values to parameters not defined in out_param, and
    % stores all the information in the struct variable out
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % out_param: struct variable containing output parameters defined by
    %            the user
    % out_param.time_start: start time (format 'dd/mm/yyyy HH:MM:SS') of
    %                       the measurements to analyse (default = first
    %                       recording)
    % out_param.time_stop: end time (format 'dd/mm/yyyy HH:MM:SS') of the
    %                      measurements to analyse (default = last
    %                      recording)
    % out_param.cell_start: index of the first cell to take into account
    %                       for the analysis (default = 1)
    % out_param.cell_stop: index of the last cell to take into account for
    %                      the analysis (default = number of cells)
    % out_param.save_format: image format to save the figures (default =
    %                        '', for display only)
    % out_param.figure_folder_name: name of the folder where the figures
    %                               are saved in png format (default =
    %                               'figures' subfolder in the deployment
    %                               folder)
    % 
    % out: struct variable containing output parameters defined by the user
    %      (see read_output_parameters help) and output data recorded by
    %      the instrument
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    % out_param.time_start
    if isfield(out_param, 'time_start')
        out.time_start = out_param.time_start;
    else
        out.time_start = dpl.time(1);
    end

    % out_param.time_stop
    if isfield(out_param, 'time_stop')
        out.time_stop = out_param.time_stop;
    else
        out.time_stop = dpl.time(end);
    end

    % out.time, out.time_start_index, out.time_stop_index
    out.time_start_index = find(out.time_start <= dpl.time, 1);
    if isempty(out.time_start_index)
        error('time_start is later than the end of the deployment')
    end
    out.time_stop_index = find(out.time_stop >= dpl.time, 1, 'last');
    if isempty(out.time_stop_index)
        error('time_stop is earlier than the start of the deployment')
    end
    out.time = dpl.time(out.time_start_index:out.time_stop_index);

    % out_param.cell_start
    if isfield(out_param, 'cell_start')
        out.cell_start = out_param.cell_start;
    else
        out.cell_start = 1;
    end

    % out_param.cell_stop
    if isfield(out_param, 'cell_stop')
        out.cell_stop = out_param.cell_stop;
    else
        out.cell_stop = dpl.nb_cells;
    end

    % out.cells, out.cell_start, out.cell_stop
    if out.cell_start < 1
        error('cell_start should be >= 1')
    end
    if out.cell_stop > dpl.nb_cells
        error('cell_stop should be <= deployment.nb_cells')
    end
    if out.cell_start > out.cell_stop
        error('cell_start should be <= cell_stop')
    end
    out.cells = out.cell_start:out.cell_stop;

    % removing cells where v is always NaN (in out.cells and in dpl.z)
    cells = [];
    for k = out.cells
        if sum(isnan(dpl.v(1, :, k))) < length(dpl.v(1, : , k))
            cells = [cells k];
        end
    end
    out.cells = cells;
    out.z = dpl.z(out.cells);

    % out_param.save_format
    if isfield(out_param, 'save_format')
        out.save_format = out_param.save_format;
    else
        out.save_format = '';
    end

    % name of the folder where the figures will be stored
    if isfield(out_param, 'figure_folder_name')
        out.figure_folder_name = out_param.figure_folder_name;
    else
        out.figure_folder_name = [dpl.folder_name '/figures/'];
    end
    if ~exist(out.figure_folder_name,'dir')
        mkdir(out.figure_folder_name);
    end