function sen = load_sen_file(dpl)
    
    % sen = LOAD_SEN_FILE(dpl) reads SEN file of the deployment and stores
    % all usefull information in the struct variable sen
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % sen: struct variable containing deployment data recorded by the
    %      instrument in the SEN file
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue


    if strcmp(dpl.data_treatment, 'raw')
        mat = load([dpl.folder_name '/' dpl.dpl_name '.sen']);
    elseif strcmp(dpl.data_treatment, 'storm')
        mat = load([dpl.folder_name '/' dpl.dpl_name '_p.sen']);
    else
        error(['load_sen_file is only defined for ' inputname(1) ' data_treatment equal to "raw" or "storm"'])
    end
    
    sen.time = zeros(size(mat, 1), 1);
    % dates
    for i = 1:size(mat, 1)
        % month
        if mat(i, 1) < 10
            month = ['0' num2str(mat(i, dpl.sen_columns.month))];
        else
            month = num2str(mat(i, dpl.sen_columns.month));
        end
        % day
        if mat(i, 2) < 10
            day = ['0' num2str(mat(i, dpl.sen_columns.day))];
        else
            day = num2str(mat(i, dpl.sen_columns.day));
        end
        % year
        year = num2str(mat(i, dpl.sen_columns.year));
        % hour
        if mat(i, 4) < 10
            hour = ['0' num2str(mat(i, dpl.sen_columns.hour))];
        else
            hour = num2str(mat(i, dpl.sen_columns.hour));
        end
        % minute
        if mat(i, 5) < 10
            minute = ['0' num2str(mat(i, dpl.sen_columns.minute))];
        else
            minute = num2str(mat(i, dpl.sen_columns.minute));
        end
        % second
        if mat(i, 6) < 10
            second = ['0' num2str(mat(i, dpl.sen_columns.second))];
        else
            second = num2str(mat(i, dpl.sen_columns.second));
        end
        % datenum
        sen.time(i) = datenum([month day year hour minute second], 'mmddyyyyHHMMSS') - dpl.time_gap / (60 * 60 * 24);
    end
    
    sen.heading = mat(:, dpl.sen_columns.heading);
    sen.pitch = mat(:, dpl.sen_columns.pitch);
    sen.roll = mat(:, dpl.sen_columns.roll);
    sen.pressure = mat(:, dpl.sen_columns.pressure);
    sen.temperature = mat(:, dpl.sen_columns.temperature);