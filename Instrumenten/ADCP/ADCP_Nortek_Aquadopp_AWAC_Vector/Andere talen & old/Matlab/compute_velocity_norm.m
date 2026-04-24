function V = compute_velocity_norm(dpl, out)

    % V = COMPUTE_DEPTH_AVERAGED_VELOCITY(dpl, out) computes the
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
    % V: matrix of size (m, n) containing the values of the velocity norm
    %    at different times and different depths
    % V(j, k): j: time index
    %          k: space index (along the profile)
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue


    v = dpl.v(:, out.time_start_index:out.time_stop_index, out.cells);
    V = zeros(length(out.time), length(out.cells));

    for j = 1:length(out.time)
        for k = 1:length(out.cells)
            V(j, k) = sqrt(v(1, j, k) ^ 2 + v(2, j, k) ^ 2 + v(3, j, k) ^ 2);
        end
    end