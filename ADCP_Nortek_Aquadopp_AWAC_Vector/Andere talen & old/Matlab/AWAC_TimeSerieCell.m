clear
close all
%AWAC

% Beam 1=East
B1=load('Herculesa660602_23092015_14112015_p.v1');
B2=load('Herculesa660602_23092015_14112015_p.v2');
B3=load('Herculesa660602_23092015_14112015_p.v3');
%cell heights
%1    0.90
%2    1.40
%3    1.90
%4    2.40
%5    2.90
%6    3.40
%7    3.90
%8    4.40



%cell 1
subplot(8,1,1)
plot(B1(:,1),'k')
hold on
plot(B2(:,1),'r')
hold on
plot(B3(:,1),'b')
ylabel('Velocity for cell 1 (m/s)')
xlim([0 6241])
%cell 2
subplot(8,1,2)
plot(B1(:,2),'k')
hold on
plot(B2(:,2),'r')
hold on
plot(B3(:,2),'b')
ylabel('Velocity for cell 2 (m/s)')
xlim([0 6241])
%cell 3
subplot(8,1,3)
plot(B1(:,3),'k')
hold on
plot(B2(:,3),'r')
hold on
plot(B3(:,3),'b')
ylabel('Velocity for cell 3 (m/s)')
xlim([0 6241])
%cell 4
subplot(8,1,4)
plot(B1(:,4),'k')
hold on
plot(B2(:,4),'r')
hold on
plot(B3(:,4),'b')
ylabel('Velocity for cell 4 (m/s)')
xlim([0 6241])
%cell 5
subplot(8,1,5)
plot(B1(:,5),'k')
hold on
plot(B2(:,5),'r')
hold on
plot(B3(:,5),'b')
ylabel('Velocity for cell 5 (m/s)')
xlim([0 6241])
%cell 6
subplot(8,1,6)
plot(B1(:,6),'k')
hold on
plot(B2(:,6),'r')
hold on
plot(B3(:,6),'b')
ylabel('Velocity for cell 6 (m/s)')
xlim([0 6241])
%cell 7
subplot(8,1,7)
plot(B1(:,7),'k')
hold on
plot(B2(:,7),'r')
hold on
plot(B3(:,7),'b')
ylabel('Velocity for cell 7 (m/s)')
xlim([0 6241])
%cell 8
% subplot(8,1,8)
% plot(B1(:,8),'k')
% hold on
% plot(B2(:,8),'r')
% hold on
% plot(B3(:,8),'b')
% xlim([0 3306])
% % ylim([-1 1])
% xlabel('Time')
% ylabel('Velocity for cell 8 (m/s)')


%K: east
%R:north
%B:vertical
 %a(a == -99) = 0;