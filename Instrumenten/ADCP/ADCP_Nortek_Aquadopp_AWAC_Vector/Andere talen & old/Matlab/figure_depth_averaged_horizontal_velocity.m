function figure_depth_averaged_horizontal_velocity(dpl, out, ebb_flood, max_velocity_display, ellipse_bounding_box_display)

    % FIGURE_TS_VELOCITY_NORM_PROFILE(dpl, out, ebb_flood,
    % max_velocity_display, ellipse_bounding_box_display) plots ... 
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % out: struct variable containing output parameters defined by the user
    %      (see read_output_parameters help) and output data recorded by
    %      the instrument
    %
    % ebb_flood: boolean to decide if we want to make the difference
    %            between ebb and flood (1) or not (0)
    %
    % max_velocity_display: maximum velocity displayed (= NaN means
    %                       maximum velocity measured)
    % 
    % ellipse_bounding_box_display: vector of size 4 containing the values
    %                               of the bounding box to display velocity
    %                               ellipses ([left right bottom top]); if
    %                               there is one NaN value in the vector,
    %                               "axis image" is used
    
    % Required packages: suplabel
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    % compute depth-averaged velocity
    vh = compute_depth_averaged_velocity(dpl, out);
    uh = vh([1 2], :);
    Uh = sqrt(uh(1, :) .^ 2 + uh(2, :) .^ 2);

    if ebb_flood
        [ebb, flood] = compute_ebb_flood(dpl, out);
    end

    time_hours = (out.time - out.time(1)) * 24;

    % figure
    figure

    % norm time series
    subplot(2, 1, 1, 'FontSize', 12)
    hold on
    if ebb_flood
        plot(time_hours, Uh .* ebb, 'b', 'LineWidth', 2)
        plot(time_hours, Uh .* flood, 'r', 'LineWidth', 2)
        legend('ebb current', 'flood current')
    else
        plot(time_hours, Uh, 'k', 'LineWidth', 2)
    end
    xlim([0 time_hours(end)])
    xlabel(['time since ' datestr(out.time(1)) ' [hours]'])
    ylabel('norm (m/s)')
    bot = 0;
    if ~isnan(max_velocity_display)
        top = max_velocity_display;
    else
        top = ceil(max(Uh) * 10) / 10;
    end
    ylim([bot top])
    box on

    % ellispe
    subplot(2, 1, 2, 'FontSize', 12)
    hold on
    if ebb_flood
        plot(uh(1, :) .* ebb, uh(2, :) .* ebb, 'b.', 'MarkerSize', 5)
        plot(uh(1, :) .* flood, uh(2, :) .* flood, 'r.', 'MarkerSize', 5)
    else
        plot(uh(1, :), uh(2, :), 'k.', 'MarkerSize', 5)
    end
    if isnan(sum(ellipse_bounding_box_display))
        axis image
    else
        axis equal
        xlim([ellipse_bounding_box_display(1) ellipse_bounding_box_display(2)]);
        ylim([ellipse_bounding_box_display(3) ellipse_bounding_box_display(4)]);
    end
    grid on
    xlabel('eastward component (m/s)')
    ylabel('northward component (m/s)')
    box on

    % main title
    [ax, h] = suplabel('Depth-averaged horizontal velocity', 't');
    set(h, 'FontSize', 14)

    % save figure
    if strcmp(out.save_format, 'png')
        saveas(gcf, [out.figure_folder_name '/depth_averaged_horizontal_velocity'], 'png');
    end