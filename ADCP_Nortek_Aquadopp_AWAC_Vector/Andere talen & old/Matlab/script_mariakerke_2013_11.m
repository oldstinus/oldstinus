% SCRIPT_MARIAKERKE_2013_11 is a script that generates the figures of the
% November 2013 Mariakerke campaign (HERCULESI:1 aquadopp and 1 AWAC; HYLASIII:1 Aquadopp)

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

% HERCULESI aquadopp
dpl_param.folder_name = 'F:/MyMatlab/CampaignAnalysis/nortek_matlab/data/2013_11_mariakerke/HerculesI/AQD8481';
dpl_param.dpl_name = 'A848103';
dpl_param.z0 = -6.5; %depth -6.5m TAW
dpl_param.reference_height = 'aquadopp';
dpl_param.vertical_dir = 'downward';
dpl_param.time_gap = 3.969 ; % to change +3.9695s (instrument later than computer)
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('13-Nov-2013 18:20:00');
out_param.time_stop = datenum('11-Dec-2013 00:00:00');
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 7, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);




%%%%%%%
% HERCULESI awac
% dpl_param.data_treatment = 'storm';
out_param.save_format = 'png';
dpl_param.data_treatment = 'storm';


dpl_param.folder_name = 'F:/MyMatlab/CampaignAnalysis/nortek_matlab/data/2013_11_mariakerke/HerculesI/AWAC';
dpl_param.dpl_name = 'W202202';
dpl_param.z0 = -6.5; %depth -6.5m TAW
dpl_param.reference_height = 'awac';
dpl_param.vertical_dir = 'upward';
dpl_param.time_gap = -1 ; % to change -1s (instrument was earlier than the computer)
dpl = load_deployment_files(dpl_param);

out_param.time_start = datenum('13-Nov-2013 00:00:00');
out_param.time_stop = datenum('11-Dec-2013 00:00:00');
%out_param.profile_days_per_subplot = 3;
out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 7, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);