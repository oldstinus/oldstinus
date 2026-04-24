% SCRIPT_MARIAKERKE_2015_09 is a script that generates the figures of the
% Sept-Nov 2015 Mariakerke campaign (HERCULESI:1 aquadopp and 1 AWAC)

% Required packages: suplabel
%
% Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
% Author: Olivier Gourgue

% test changement

close all
clear all

% dpl_param.data_treatment = 'storm';
out_param.save_format = 'png';
dpl_param.data_treatment = 'storm';


% HERCULESI awac
% dpl_param.data_treatment = 'storm';
out_param.save_format = 'png';
dpl_param.data_treatment = 'storm';


dpl_param.folder_name = 'F:\MyMatlab\CampaignAnalysis\nortek_matlab\data\2015_09_mariakerke\M1_Hercules\AWAC\Processed23092015_14112015';
dpl_param.dpl_name = 'Herculesa660602_23092015_14112015';
dpl_param.z0 = -6.5; %depth -6.5m TAW
dpl_param.reference_height = 'awac';
dpl_param.vertical_dir = 'upward';
dpl_param.time_gap = -14 ; % to change -14s (instrument was earlier than the computer)
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('23-Sept-2015 00:01:01');
out_param.time_stop = datenum('14-Nov-2015 00:01:01');
%out_param.profile_days_per_subplot = 3;
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 7, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);