function [ebb, flood] = compute_ebb_flood(dpl, out)

    % [ebb, flood] = COMPUTE_EBB_FLOOD(dpl, out) determines for each time
    % step if it is ebb or flood
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % out: struct variable containing output parameters defined by the user
    %      (see read_output_parameters help) and output data recorded by
    %      the instrument
    %
    % ebb and flood: vector od size m with
    %   ebb(i) = 1 and flood(i) = NaN if it is ebb courrent at time step i
    %   ebb(i) = NaN and flood(i) = 1 if it is flood current at time step i
    %
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue
    
    
    v = dpl.v(:, out.time_start_index:out.time_stop_index, out.cells);

    vh = compute_depth_averaged_velocity(dpl, out);
    uh = vh([1 2], :);
    Uh = sqrt(uh(1, :) .^ 2 + uh(2, :) .^ 2);

    % determine the time indices of slack waters (when velocity is minimum)
    min_ind = 1;
    for i = 1:length(Uh)
        min_loc = min(Uh(max(1, i - 24) : min(length(Uh), i + 24)));
        if Uh(i) == min_loc
            min_ind = [min_ind i];
        end
    end
    min_ind = [min_ind length(Uh)];

    % determine the number of nan cells at each time step (to have an idea of 
    % the water elevation
    for i = 1:length(out.time)
        nan_cells(i) = sum(isnan(v(1, i, :)));
    end

    % determine if the current is ebb or flood at each time step
    mean_nan_cells = mean(nan_cells);
    ebb = ones(1, length(out.time));
    flood = ones(1, length(out.time));
    for i = 1:length(min_ind) - 1
        % ebb
        if mean(nan_cells(min_ind(i) : min_ind(i + 1))) > mean_nan_cells
            flood(min_ind(i) + 1 : min_ind(i + 1) - 1) = NaN;
        else % flood
            ebb(min_ind(i) + 1 : min_ind(i + 1) - 1) = NaN;
        end
    end