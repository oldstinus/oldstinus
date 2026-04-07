% SCRIPT_MARIAKERKE_2013_08 is a script that generates the figures of the
% August 2013 Mariakerke campaign (2 aquadopp and 1 AWAC)

% Required packages: suplabel
%
% Developed at Flanders Hydraulics Research (FHR), Antwerp, Belgium
% Author: Olivier Gourgue

% test changement

close all
clear all

% dpl_param.data_treatment = 'storm';
out_param.save_format = 'png';

% aquadopp
dpl_param.folder_name = '../data/2013_08_mariakerke/A847102';
dpl_param.dpl_name = 'A847102';
% dpl_param.z0 = -4.5;
dpl_param.reference_height = 'aquadopp';
dpl_param.vertical_dir = 'downward';
dpl_param.time_gap = 0; % to change 
dpl = load_deployment_files(dpl_param);

out = read_output_parameters(dpl, out_param);

figure_ts_instrument_orientation(dpl, out);
figure_ts_depth_averaged_velocity_components(dpl, out);
figure_ts_velocity_norm_profile(dpl, out, 7, NaN, NaN, NaN);
figure_depth_averaged_horizontal_velocity(dpl, out, 0, NaN, [NaN NaN NaN NaN]);