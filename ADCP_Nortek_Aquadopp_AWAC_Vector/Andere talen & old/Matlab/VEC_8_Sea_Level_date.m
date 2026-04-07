% Sea Level by AWAC vs Vec

day = find((wave(:,1)==10&(wave(:,2)==04))); % month 10 day 04

lev = sea_level(day);
t1 = [1:1200:14400];

see = zeros(14400,1);

for i = 1:14400
    see(i) = NaN;
end

see(t1) = lev;


figure

t= [0:14400];
plot(surface_vec)
axis([0 14400 0 12])

hold on

plot(see, 'r:+')

