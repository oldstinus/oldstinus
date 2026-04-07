function hdr = load_hdr_file(dpl)

    % hdr = LOAD_HDR_FILE(dpl) reads HDR file of the deployment and stores
    % all usefull information in the struct variable hdr
    %
    % dpl: struct variable containing deployment parameters defined by the
    %      user (see read_deployment_parameters help) and deployment
    %      data recorded by the instrument
    %
    % hdr: struct variable containing deployment data recorded by the
    %      instrument in the HDR file
    
    % Required packages: -
    %
    % Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
    % Author: Olivier Gourgue

    
    fid = fopen([dpl.folder_name '/' dpl.dpl_name '.hdr']);
    tline = fgetl(fid);

    while ischar(tline)

        keyword = 'Number of measurements';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            cline = regexp(tline, '  ', 'split');
            hdr.nb_measurements = str2num(cline{end});
        end

        keyword = 'Time of first measurement';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            cline = regexp(tline, '  ', 'split');
            if length(cline{end}) > 11
                hdr.time_start = datenum(cline{end}, 'dd/mm/yyyy HH:MM:SS');
            else
                hdr.time_start = datenum(cline{end}, 'dd/mm/yyyy');
            end
        end

        keyword = 'Time of last measurement';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            cline = regexp(tline, '  ', 'split');
            hdr.time_stop = datenum(cline{end}, 'dd/mm/yyyy HH:MM:SS');
        end

        keyword_1 = 'Profile interval';
        keyword_2 = 'Measurement/Burst interval';
        if strcmp(tline(1:min(length(keyword_1), end)), keyword_1) || strcmp(tline(1:min(length(keyword_2), end)), keyword_2)
            cline = regexp(tline, '  ', 'split');
            cline = regexp(cline{end}, ' ', 'split');
            if strcmp(cline{end}, 'sec')
                hdr.dt = str2num(cline{1});
            else
                disp('load_hdr_file only reads "Profile interval" or "Measurement/Burst interval" in seconds; other units to be implemented')
            end
        end

        keyword = 'Number of cells';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            cline = regexp(tline, '  ', 'split');
            hdr.nb_cells = str2num(cline{end});
        end

        keyword = 'Coordinate system';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            cline = regexp(tline, '  ', 'split');
            cline = regexp(cline{end}, ' ', 'split');
            hdr.coordinate_system = cline{end};
        end

        keyword = 'Current profile cell center distance from head (m)';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            tline = fgetl(fid);
            while ~strcmp(tline, '')
                tline = fgetl(fid);
                cline = regexp(tline, '   ', 'split');
                if strcmp(dpl.vertical_dir, 'upward')
                    hdr.z(str2num(cline{1})) = dpl.z0 + str2num(cline{end});
                else
                    hdr.z(str2num(cline{1})) = dpl.z0 - str2num(cline{end});
                end
            end
        end
        
        keyword = 'Current profile cell center distances from transducer head.';
        if strcmp(tline(1:min(length(keyword), end)), keyword)
            tline = fgetl(fid);
            tline = fgetl(fid);
            tline = fgetl(fid);
            while ~strcmp(tline, '')
                tline = fgetl(fid);
                cline = regexp(tline, '   ', 'split');
                if strcmp(dpl.vertical_dir, 'upward')
                    hdr.z(str2num(cline{1})) = dpl.z0 + str2num(cline{end});
                else
                    hdr.z(str2num(cline{1})) = dpl.z0 - str2num(cline{end});
                end
            end
        end
        
        keyword = '.sen';
        if strcmp(tline(max(1, end - 4):end - 1), keyword)
            tline = fgetl(fid);
            while ~strcmp(tline, '')
                cline = regexp(tline, '  ', 'split');
                if strcmp(strtrim(cline{2}), 'Month')
                    hdr.sen_columns.month = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Day')
                    hdr.sen_columns.day = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Year')
                    hdr.sen_columns.year = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Hour')
                    hdr.sen_columns.hour = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Minute')
                    hdr.sen_columns.minute = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Second')
                    hdr.sen_columns.second = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Error code')
                    hdr.sen_columns.error_code = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Status code')
                    hdr.sen_columns.status_code = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Battery voltage')
                    hdr.sen_columns.battery_voltage = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Soundspeed')
                    hdr.sen_columns.soundspeed = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Heading')
                    hdr.sen_columns.heading = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Pitch')
                    hdr.sen_columns.pitch = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Roll')
                    hdr.sen_columns.roll = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Pressure')
                    hdr.sen_columns.pressure = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Temperature')
                    hdr.sen_columns.temperature = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Analog input 1')
                    hdr.sen_columns.analog_input_1 = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Analog input 2')
                    hdr.sen_columns.analog_input_2 = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Burst counter')
                    hdr.sen_columns.burst_counter = str2num(cline{1});
                elseif strcmp(strtrim(cline{2}), 'Ensemble counter')
                    hdr.sen_columns.ensemble_counter = str2num(cline{1});
                end
                tline = fgetl(fid);
            end
        end
        
        keyword = '.v1';
        if strcmp(tline(max(1, end - 3):end - 1), keyword)
            tline = fgetl(fid);
            while ~strcmp(tline, '')
                cline = regexp(tline, '  ', 'split');
                if strcmp(strtrim(cline{2}), 'Velocity Cell 1 (Beam1|X|East)')
                    hdr.v_columns.cell_1 = str2num(cline{1});
                end
                tline = fgetl(fid);
            end
        end

        tline = fgetl(fid);
    end

    fclose(fid);