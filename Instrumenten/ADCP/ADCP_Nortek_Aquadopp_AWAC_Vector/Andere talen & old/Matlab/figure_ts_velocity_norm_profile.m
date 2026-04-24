function figure_ts_velocity_norm_profile(dpl, out, days_per_subplot, min_depth_display, max_depth_display, max_velocity_display)

    % FIGURE_TS_VELOCITY_NORM_PROFILE(dpl, out, days_per_subplot,
    % min_depth_display, max_depth_display, max_velocity_display) plots
    % time series of the 3 components (eastward, northward, upward) of the
    % depth-averaged velocity
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % out: struct variable containing output parameters defined by the user
    %      (see read_output_parameters help) and output data recorded by
    %      the instrument
    %
    % days_per_subplot: number of days displayed on each subplot
    %
    % min_depth_display: minimum depth displayed (= NaN means minimum depth
    %                    measured)
    % 
    % max_depth_display: maximum deplth displayed (= NaN means maximum
    %                    depth measured)
    %
    % max_velocity_display: maximum velocity displayed (= NaN means maximum
    %                       velocity measured)
    
    % Required packages: suplabel
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue


    % compute velocity norm
    V = compute_velocity_norm(dpl, out);

    % time definitions
    time_start = floor(out.time(1));
    time_stop = ceil(out.time(end));
    nb_subplot = ceil((time_stop - time_start) / days_per_subplot);
    
    % minimum/maximum depth/velocity displayed
    if isnan(min_depth_display)
        min_depth_display = min(out.z);
    end
    if isnan(max_depth_display)
        max_depth_display = max(out.z);
    end
    if isnan(max_velocity_display)
        max_velocity_display = max(max(V));
    end

    % figure
    figure

    for i = 1:nb_subplot
        subplot(nb_subplot, 1, i, 'FontSize', 12)
        surf(out.time, out.z, V')
        shading interp
        view(2)
        colorbar
        time_start_local = time_start + (i - 1) * days_per_subplot;
        time_stop_local = time_start + i * days_per_subplot;
        xlim([time_start_local time_stop_local])
        ylim([min_depth_display max_depth_display])
        box on
        grid off
        time_tick = [time_start_local : time_stop_local];
        set(gca, 'XTick', time_tick)
        set(gca, 'XTickLabel', datestr(time_tick, 'dd-mmm'))
       %ylabel({'height above sensor [m]' ; [dpl.reference_height ' [m]']}, 'FontSize', 10)
          ylabel({'height above sensor (m)'}, 'FontSize', 10)
        if max_velocity_display > 0
            caxis([0 max_velocity_display])
        end
    end

    [ax, h] = suplabel('Velocity norm [m/s]', 't');
    set(h, 'FontSize', 14)

    % to have each subplot like a standard (3, 1) subplot
    pos = get(gcf, 'Position');
    pap = get(gcf, 'PaperPosition');
    set(gcf, 'Position', [pos(1) 0 pos(3) pos(4) / 3 * nb_subplot])
    set(gcf, 'PaperPosition', [pap(1) pap(2) pap(3) pap(4) / 3 * nb_subplot])

    % to have a nice figure
    set(gcf, 'Renderer', 'Zbuffer')

    % save figure
    if strcmp(out.save_format, 'png')
        saveas(gcf, [out.figure_folder_name '/ts_velocity_norm_profile'], 'png');
    end