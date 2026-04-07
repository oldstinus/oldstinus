function vh = compute_depth_averaged_velocity(dpl, out)

    % vh = COMPUTE_DEPTH_AVERAGED_VELOCITY(dpl, out) computes the
    % depth-averaged velocity components from the vertical velocity profile
    % measurements 
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % out: struct variable containing output parameters defined by the user
    %      (see read_output_parameters help) and output data recorded by
    %      the instrument
    %
    % vh: matrix of size (3, m) containing the values of the depth-averaged
    %     velocity components at different times
    % vh(i, j): i: index of the depth-averaged velocity component (1 =
    %              eastward, 2 = northward, 3 = upward)
    %           j: time index
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    v = dpl.v(:, out.time_start_index:out.time_stop_index, out.cells);
    vh = zeros(3, length(out.time));

    for j = 1:length(out.time)
            % vectors containing the value of each component along the profile
            v1 = v(1, j, :);
            v2 = v(2, j, :);
            v3 = v(3, j, :);

            % removing the NaN data
            v1 = v1(find(~isnan(v1)));
            v2 = v2(find(~isnan(v2)));
            v3 = v3(find(~isnan(v3)));

            % integrate along the profile
            % we may use "mean" because cells are of equal length and NaN are
            % only at the end of the profile (bottom or surface)
            vh(1, j) = mean(v1);
            vh(2, j) = mean(v2);
            vh(3, j) = mean(v3);
    end