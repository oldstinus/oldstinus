% SCRIPT_BLANKENBERGE_2012 is a script that generates the figures of the
% June 2012 Blankenberge campaign (2 aquadopp)

% Required packages: suplabel
%
% Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
% Author: Olivier Gourgue

close all
clear all

out_param.save_format = 'png';

% aquadopp_1
dpl_param.folder_name = '../data/2012_06_blankenberge/aquadopp_1';
dpl_param.dpl_name = 'Averti01';
dpl_param.z0 = -5.3;
dpl_param.vertical_dir = 'downward';
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('5-Jun-2012 11:10:00');
out_param.cell_stop = 14;
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 5, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);

% aquadopp_2
dpl_param.folder_name = '../data/2012_06_blankenberge/aquadopp_2';
dpl_param.dpl_name = 'Ahoriz01';
dpl_param.z0 = -4.6;
dpl_param.vertical_dir = 'downward';
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('5-Jun-2012 11:30:00');
out_param.time_stop = datenum('26-Jun_2012 08:25:00');
out_param.cell_stop = 17;
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 5, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);