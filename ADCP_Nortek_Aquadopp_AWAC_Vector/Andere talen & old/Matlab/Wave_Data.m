% Waves Speed and Direction, Max Wave from AWAC


wave = load('Herculesa660602_23092015_14112015.wap');

day = find((wave(:,1)==10&(wave(:,2)==14))); % month 10 day 14
%
%
b = wave_max(day);
% 
c = current_direction(day);
% 
d = current_speed(day);
% 
f = mean_direction(day);
% 
g = mean_period(day);   % period
% 
h = signif_height(day);
% 
l = peak_period(day);
% 
m = peak_dir(day);
%
%
t1 = [1:1200:14400];
% 

max_wave = zeros(14400,1);

curr_d = zeros(14400,1); 

curr_sp = zeros(14400,1);

wave_d = zeros(14400,1);

wave_p =zeros(14400,1); % period

wave_sig = zeros(14400,1);    

peak_p = zeros(14400,1); 

peak_d = zeros(14400,1);

for i = 1:14400
    
    max_wave(i) = NaN;
    curr_d(i) = NaN;
    curr_sp(i) = NaN;
    wave_d(i) = NaN;
    wave_p(i) = NaN;
    wave_sig(i) = NaN;
    peak_p(i) = NaN;
    peak_d(i) = NaN;
    
end


    max_wave(t1) = b;
    curr_d(t1) = c;
    curr_sp(t1) = d;
    wave_d(t1) = f;
    wave_p(t1) = g;
    wave_sig(t1) = h;
    peak_p(t1) = l;
    peak_d(t1) = m;



    