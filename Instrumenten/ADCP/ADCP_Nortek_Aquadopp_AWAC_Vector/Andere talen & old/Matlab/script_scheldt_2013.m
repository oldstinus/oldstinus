% SCRIPT_SCHELDT_2013 is a script that generates the figures of the
% July 2013 Scheldt campaign (1 aquadopp)

% Required packages: suplabel
%
% Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
% Author: Olivier Gourgue

close all
clear all

% --------------------- %
% deployment parameters %
% --------------------- %

% folder name (with path) where the aquadopp files are stored
dpl_param.folder_name = '../data/2013_07_scheldt/aquadopp';
% name of the aquadopp files
dpl_param.dpl_name = 'ALNOT02';
% name of the reference height ('aquadopp' means that vertical position 
% will be expressed as height above the aquadopp, but it could be 'TAW', 
% 'NAP', ...) 
dpl_param.reference_height = 'aquadopp';
% vertical position of the aquadopp relative to the reference height
dpl_param.z0 = 0;
% orientation of the aquadopp (could be 'upward' or 'downward')
dpl_param.vertical_dir = 'upward';
% to use the data processed by Storm
dpl_param.data_treatment = 'storm';



% -------------------------- %
% load aquadopp output files %
% -------------------------- %

dpl = load_deployment_files(dpl_param);


% ----------------- %
% output parameters %
% ----------------- %

% if you do not define time_start and time_stop, it will take the first and
% last measurements
out_param.time_start = datenum('13-Jul-2013 00:00:00');
out_param.time_stop = datenum('13-Aug-2013 15:30:00');
% file format to save the figures (if commented, the figures will only be
% displayed on the screen)
out_param.save_format = 'png';


% ---------------------- %
% read output parameters %
% ---------------------- %

out = read_output_parameters(dpl, out_param);


% ---------------- %
% generate figures %
% ---------------- %

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 7, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);