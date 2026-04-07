function dpl = read_deployment_parameters(dpl_param)

    % dpl = READ_DEPLOYMENT_PARAMETERS(dpl_param) reads deployment
    % parameters defined in dpl_param, checks if values are corrects,
    % assigns default values to parameters not defined in dpl_param, and
    % stores all the information in the struct variable dpl
    %
    % dpl_param: struct variable containing deployment parameters defined
    %            by the user
    % dpl_param.folder_name: name of the folder (with path) containing the
    %                        deployment output files (required)
    % dpl_param.dpl_name: name of the deployment (required)
    % dpl_param.z0: depth of the instrument during deployment (default = 0)
    % dpl_param.vertical_dir: orientation of the instrument (can be
    %                         'upward' or 'downward', default = 'upward')
    % dpl_param.vertical_datum: vertical datum for water elevation (default
    %                           = 'TAW')
    % dpl_param.data_treatment: treatment of the data (can be 'raw' for no
    %                           treatment or 'storm' for treatment using
    %                           Nortek storm software, default is 'raw')
    % dpl_param.time_gap: time difference (in seconds) between instrument
    %                     and reality (positive if instrument time is
    %                     higher than real time)
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    % dpl_param.folder_name
    if isfield(dpl_param, 'folder_name')
        dpl.folder_name = dpl_param.folder_name;
    else
        error([inputname(1) '.folder_name must be defined'])
    end
    
    % dpl_param.dpl_name
    if isfield(dpl_param, 'dpl_name')
        dpl.dpl_name = dpl_param.dpl_name;
    else
        error([inputname(1) '.dpl_name must be defined'])
    end

    % dpl_param.z0
    if isfield(dpl_param, 'z0')
        dpl.z0 = dpl_param.z0;
    else
        dpl.z0 = 0;
    end

    % dpl_param.vertical_dir
    if isfield(dpl_param, 'vertical_dir')
        if strcmp(dpl_param.vertical_dir, 'upward') || strcmp(dpl_param.vertical_dir, 'downward')
            dpl.vertical_dir = dpl_param.vertical_dir;
        else
            error([inputname(1) '.vertical_dir should be "upward" or "downward", not "' dpl_param.vertical_dir '"'])
        end
    else
        dpl.vertical_dir = 'upward';
    end

    % dpl_param.vertical_datum
    if isfield(dpl_param, 'reference_height')
        dpl.reference_height = dpl_param.reference_height;
    else
        dpl.reference_height = 'TAW';
    end

    % dpl_param.data_treatment
    if isfield(dpl_param, 'data_treatment')
        dpl.data_treatment = dpl_param.data_treatment;
    else
        dpl.data_treatment = 'raw';
    end
    
    % dpl_param.time_gap
    if isfield(dpl_param, 'time_gap')
        dpl.time_gap = dpl_param.time_gap;
    else
        dpl.time_gap = 0;
    end