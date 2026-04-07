addpath(genpath ('p:\00_128-VooroevGeulSu\9_Rapportering\Hydrodynamics_vs_Sediments\Francesca\Matlab Analysis\Poster\Hercules\20151004'));

wave = load('Herculesa660602_23092015_14112015.wap');

date_campaign = wave(:,1:6);
signif_height = wave(:,8);
wave_H3 = wave(:,9);
wave_max = wave(:,11);

wave_mean = wave(:,12);

mean_period = wave(:,16);
mean_period = wave(:,13);
period_T3 = wave(:,16);
peak_period = wave(:,14);
max_period_Tmax = wave(:,18);
mean_direction = wave(:,21);
mean_pressure = wave(:,23);
current_speed = wave(:,29);
current_direction = wave(:,30);

sea_level = 1.40+((mean_pressure)*1.019716);