function figure_ts_instrument_orientation(dpl, out)

    % FIGURE_TS_INSTRUMENT_ORIENTATION(dpl, out) plots time series of
    % instrument orientation angles (heading, pitch and roll)
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

    time_hours = (out.time - out.time(1)) * 24;

    % figure
    figure('Units','normalized','position',[0 0 1 1])

    % eastward component
    subplot(3, 1, 1, 'FontSize', 12)
    plot(time_hours, dpl.heading(out.time_start_index:out.time_stop_index), 'k', 'LineWidth', 2)
    ylabel('heading (°)')
    axis tight

    % northward component
    subplot(3, 1, 2, 'FontSize', 12)
    plot(time_hours, dpl.pitch(out.time_start_index:out.time_stop_index), 'k', 'LineWidth', 2)
    ylabel('picth (°)')
    axis tight

    % upward component
    subplot(3, 1, 3, 'FontSize', 12)
    plot(time_hours, dpl.roll(out.time_start_index:out.time_stop_index), 'k', 'LineWidth', 2)
    ylabel('roll (°)')
    axis tight
    xlabel(['time since ' datestr(out.time(1)) ' [hours]'])

    % main title
    [ax, h] = suplabel('Instrument orientation', 't');
    set(h, 'FontSize', 14)

    % save figure
    if strcmp(out.save_format, 'png')
        saveas(gcf, [out.figure_folder_name '/ts_intrument_orientation'], 'png');
    end