figure

dir_aw = [dir8 dir6 dir5 dir4 dir3 dir2 dir1];
y = dir_aw;

startdate = datenum('23-Sep-2015 00:00:01');
enddate = datenum('23-Sep-2015 23:00:01');
sdate = linspace(startdate, enddate, 120);
%
x = sdate;
str = datestr(x);
%
subplot(3,1,1)
plot(x,y,'.')
xlim([startdate enddate]) 
ylim([0 360])
ylabel('T (s)')
% 
NumTicks = 24;
L = get(gca, 'xlim');
set(gca, 'Xtick', linspace(L(1),L(2),NumTicks));
set(gca, 'XtickLabel',' ');
%
%
datetick('x','HH:MM','keeplimits','keepticks');

% hold on
%
dir_vec = alfa_dir_date;
y = dir_vec;
%
startdate = datenum('23-Sep-2015 00:20:00');
enddate = datenum('23-Sep-2015 23:20:00');
sdate = linspace(startdate, enddate, 14400);
%
x = sdate;
str = datestr(x);

subplot(3,1,3)
plot(x,y,'.')
xlim([startdate enddate]) 
ylim([0 360])
ylabel('vel dir (\circ)')


NumTicks = 24;
L = get(gca, 'xlim');
set(gca, 'Xtick', linspace(L(1),L(2),NumTicks));
%
%
datetick('x','HH:MM','keeplimits','keepticks');
set(gca,'XMinorTick','on');
% figure
% 
% subplot(4,1,1)
% plot(dir4)
% ylim([0 360])
% 
% subplot(4,1,2)
% plot(dir3)
% ylim([0 360])
% 
% subplot(4,1,3)
% plot(dir2)
% ylim([0 360])
% 
% subplot(4,1,4)
% plot(dir1)
% ylim([0 360])

% figure

dir_adp = [dir2b dir4b dir6b  dir8b dir10];
y = dir_adp;

startdate = datenum('23-Sep-2015 00:00:01');
enddate = datenum('23-Sep-2015 23:00:01');
sdate = linspace(startdate, enddate, 144);
%
x = sdate;
str = datestr(x);
%
subplot(3,1,2)
plot(x,y,'.')
xlim([startdate enddate]) 
ylim([0 360])
ylabel('T (s)')
% 
NumTicks = 24;
L = get(gca, 'xlim');
set(gca, 'Xtick', linspace(L(1),L(2),NumTicks));
set(gca, 'XtickLabel',' ');
%
%
datetick('x','HH:MM','keeplimits','keepticks');


