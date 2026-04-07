clear
close all
temp=load('Herculesa660602_23092015_14112015_p.sen');

temperature=temp(:,15);
pressure=temp(:,14);

time=datenum(temp(:,3),temp(:,1),temp(:,2),temp(:,4),temp(:,5),0);

figure(1)
plot(time,temperature,'k')
datetick('x','dd-mm');
ylabel('Temperature (°C)','fontsize',14)

figure(2)
plot(time,pressure,'k')
datetick('x','dd-mm');
ylabel('Pressure(dbar)','fontsize',14)