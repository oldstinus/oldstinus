function figure_ts_depth_averaged_velocity_components(dpl, out)

    % FIGURE_TS_DEPTH_AVERAGED_VELOCITY_COMPONENTS(dpl, out) plots time
    % series of depth-averaged velocity components (eastward, northward,
    % upward)
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % out: struct variable containing output parameters defined by the user
    %      (see read_output_parameters help) and output data recorded by
    %      the instrument
    
    % Required packages: suplabel
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    % compute depth-averaged velocity
    vh = compute_depth_averaged_velocity(dpl, out);

    time_hours = (out.time - out.time(1)) * 24;

    % figure
    figure

    % eastward component
    subplot(3, 1, 1, 'FontSize', 12)
    plot(time_hours, vh(1, :), 'k', 'LineWidth', 2)
    xlim([0 time_hours(end)])
    title('Eastward component')
    ylabel('(m/s)')
    bot = floor(min(vh(1, :)) * 10) / 10;
    top = ceil(max(vh(1, :)) * 10) / 10;
    ylim([bot top])

    % northward component
    subplot(3, 1, 2, 'FontSize', 12)
    plot(time_hours, vh(2, :), 'k', 'LineWidth', 2)
    xlim([0 time_hours(end)])
    title('Northward component')
    ylabel('(m/s)')
    bot = floor(min(vh(2, :)) * 10) / 10;
    top = ceil(max(vh(2, :)) * 10) / 10;
    ylim([bot top])

    % upward component
    subplot(3, 1, 3, 'FontSize', 12)
    plot(time_hours, vh(3, :), 'k', 'LineWidth', 2)
    xlim([0 time_hours(end)])
    title('Upward component')
    ylabel('(m/s)')
    xlabel(['time since ' datestr(out.time(1)) ' [hours]'])
    bot = floor(min(vh(3, :)) * 100) / 100;
    top = ceil(max(vh(3, :)) * 100) / 100;
    ylim([bot top])

    % main title
    [ax, h] = suplabel('Depth-averaged velocity', 't');
    set(h, 'FontSize', 14)

    % save figure
    if strcmp(out.save_format, 'png')
        saveas(gcf, [out.figure_folder_name '/ts_depth_averaged_velocity_components'], 'png');
    end