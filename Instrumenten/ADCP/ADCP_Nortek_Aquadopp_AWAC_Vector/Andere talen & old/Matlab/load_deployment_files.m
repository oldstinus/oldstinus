function dpl = load_deployment_files(dpl_param)

    % dpl = LOAD_DEPLOYMENT_FILES(dpl_param) loads data recorded by a
    % NORTEK instrument (Aquadopp Profiler and AWAC have been tested),
    % according to the deployment parameters defined by the user in
    % dpl_param, and stores them in the struct variable dpl
    %
    % dpl_param: struct variable containing deployment parameters defined
    %            by the user (see read_deployment_parameters help)
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    % load deployment parameters
    dpl = read_deployment_parameters(dpl_param);

    % load informations from the HDR file
    disp('read HDR file')
    hdr = load_hdr_file(dpl);
    dpl.nb_measurements = hdr.nb_measurements;
    dpl.nb_cells = hdr.nb_cells;
    dpl.coordinate_system = hdr.coordinate_system;
    dpl.z = hdr.z;
    dpl.sen_columns = hdr.sen_columns;
    dpl.v_columns = hdr.v_columns;

    % load informations and measurement data from the SEN file
    disp('read SEN file')
    sen = load_sen_file(dpl);
    dpl.time = sen.time;
    dpl.pressure = sen.pressure;
    dpl.heading = sen.heading;
    dpl.pitch = sen.pitch;
    dpl.roll = sen.roll;

    % load velocity measurement from the V* files
    disp('read V* files')
    dpl.v = load_v_files(dpl);