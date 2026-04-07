% SCRIPT_BLANKENBERGE_2013 is a script that generates the figures of the
% June 2013 Blankenberge campaign (1 aquadopp and 2 AWAC)

% Required packages: suplabel
%
% Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
% Author: Olivier Gourgue

close all
clear all

dpl_param.data_treatment = 'storm';
out_param.save_format = 'png';

% aquadopp
dpl_param.folder_name = '../data/2013_06_blankenberge/aquadopp';
dpl_param.dpl_name = 'Q847902_5July_2013';
dpl_param.z0 = -4.5;
dpl_param.reference_height = 'TAW';
dpl_param.vertical_dir = 'upward';
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('19-Jun-2013 10:00:00');
out_param.time_stop = datenum('4-Jul-2013 11:30:00');
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 3, -4.2, 3.5, 1.6654);
figure_depth_averaged_horizontal_velocity(dpl, out, 1, 1.6, [-0.9 1.4 -0.7 1.1]);

% awac sea
dpl_param.folder_name = '../data/2013_06_blankenberge/awac_sea';
dpl_param.dpl_name = 'W660601_AWAC_sea_5_july_2013';
dpl_param.z0 = -5.1;
dpl_param.reference_height = 'TAW';
dpl_param.instrument_orientation = 'upward';
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('19-Jun-2013 10:05:00');
out_param.time_stop = datenum('4-Jul-2013 10:40:00');
out_param.profile_days_per_subplot = 3;
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 3, -4.2, 3.5, 1.6654);
figure_depth_averaged_horizontal_velocity(dpl, out, 1, 1.6, [-0.9 1.4 -0.7 1.1]);

% awac harbor
dpl_param.folder_name = '../data/2013_06_blankenberge/awac_harbor';
dpl_param.dpl_name = 'W659404';
dpl_param.z0 = -1.2;
dpl_param.reference_height = 'TAW';
dpl_param.instrument_orientation = 'upward';
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('20-Jun-2013 13:00:00');
out_param.time_stop = datenum('5-Jul-2013 11:20:00');
out_param.profile_days_per_subplot = 3;
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 3, -4.2, 3.5, 1.6654);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, 1.6, [-0.9 1.4 -0.7 1.1]);