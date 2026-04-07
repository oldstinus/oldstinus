function v_enu = load_v_files(dpl)

    % v_enu = LOAD_V_FILES(dpl) reads V* files of the deployment
    % (containing velocity measurements), changes the coordinate system in
    % (northward, eastward, upward) if not done yet, and stores  
    % all usefull information in the matrix v_enu
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % v_enu: matrix of size (3, m, n) containing the values of the velocity
    %        profile at different times
    % v_enu(i, j, k): i: index of the velocity component (1 = eastward, 2 =
    %                    northward, 3 = upward)
    %                 j: time index
    %                 k: space index (along the profile)
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue
    
    
    v = zeros(3, length(dpl.time), length(dpl.z));
    
    for i = 1:3
        if strcmp(dpl.data_treatment, 'raw')
            mat = load([dpl.folder_name '/' dpl.dpl_name '.v' num2str(i)]);
        elseif strcmp(dpl.data_treatment, 'storm')
            mat = load([dpl.folder_name '/' dpl.dpl_name '_p.v' num2str(i)]);
        else
            error(['load_v_files is only defined for ' inputname(1) ' data_treatment equal to "raw" or "storm"'])
        end
    
        v(i, :, :) = mat(:, dpl.v_columns.cell_1:end);
    end
    
    if strcmp(dpl.data_treatment, 'storm')
        ind = (v == -99);
        v(ind) = NaN;
    end
    

    if strcmp(dpl.vertical_dir, 'downward')
        v(2, :, :) = - v(2, :, :);
        v(3, :, :) = - v(3, :, :);
    end

    if strcmp(dpl.coordinate_system, 'ENU')
       v_enu = v;

    elseif strcmp(dpl.coordinate_system, 'XYZ')

        for j = 1:dpl.nb_measurements

            heading = dpl.heading(j) - 90;
            pitch = dpl.pitch(j);
            roll = dpl.roll(j);
            % heading matrix
            H = [cosd(heading) sind(heading) 0; ...
                 -sind(heading) cosd(heading) 0; ...
                 0 0 1];
            % tilt matrix
            T = [cosd(pitch) -sind(pitch)*sind(roll) -cosd(roll)*sind(pitch); ...
                 0 cosd(roll) -sind(roll); ...
                 sind(pitch) sind(roll)*cosd(pitch) cosd(pitch)*cosd(roll)];

            for k = 1:dpl.nb_cells
                % transformation
                v_rot = H * T * [v(1, j, k); v(2, j, k); v(3, j, k)];
                v_enu(1, j, k) = v_rot(1); 
                v_enu(2, j, k) = v_rot(2);
                v_enu(3, j, k) = v_rot(3);
            end
        end

    else
       error(['the coordinate system transformation ' dpl.coordinate_system ' to ENU is not implemented yet'])

    end