
%WorkDir='/projects/13_131_bevaarbozs/2050_Reference_QE_A0CH/2050_Reference_QE_A0CH_newBC3_kw';

clear all;
%% 1  %%%%%%%%%    Load data *.dat file       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%addpath(genpath ('p:\00_128-VooroevGeulSu\9_Rapportering\Hydrodynamics_vs_Sediments\Francesca\MatLab\20151014'));
%addpath(genpath ('p:\00_128-VooroevGeulSu\9_Rapportering\Hydrodynamics_vs_Sediments\Francesca\MatLab\20151014'));
addpath(genpath ('p:\00_128-VooroevGeulSu\9_Rapportering\Hydrodynamics_vs_Sediments\Francesca\Mariakerke Inside\Hercules\20151013'));

%datadir = 'p:\00_128-VooroevGeulSu\3_Uitvoering\Data\Campaigns\20150923_20151114\M1_Hercules\Vector';
files = dir('vec*.dat');
for i=1:length(files)
    eval(['load ' files(i).name ' -ascii']);
    fnameS (i) = {files(i).name};
end
fnameS = fnameS';
% % % 
% % % for i = 1:length(fnameS)
% % % %vec = load('VEC01501505.dat');
% % % file (i) = celltochar('fnameS(i)');  
% % % vel_E = file(:,3);
% % % vel_N = file(:,4);
% % % vel_U = file(:,5);
% % % press = file(:,15);
% % % end


% %entire simulation period (without spin up period)
% datenumStart_QN=datenum('Aug-12-2050 22:00:00');
% datenumEnd_QN=datenum('Nov-12-2050 21:00:00');

%%%%%%%%%%%%%%%%     Load data *.dat files      %%%%%%%%%%%%%%%%%%%%%%%%%%%

vec = load('VEC01501505.dat');
vel_E_1 = vec(:,3);
vel_N_1 = vec(:,4);
vel_U_1 = vec(:,5);
press_1 = vec(:,15);

clear vec

vec = load('VEC01501506.dat');
vel_E_2= vec(:,3);
vel_N_2= vec(:,4);
vel_U_2= vec(:,5);
press_2 = vec(:,15);

clear vec

vec = load('VEC01501507.dat');
vel_E_3 = vec(:,3);
vel_N_3 = vec(:,4);
vel_U_3 = vec(:,5);
press_3 = vec(:,15);

clear vec

vec = load('VEC01501508.dat');
vel_E_4 = vec(:,3);
vel_N_4 = vec(:,4);
vel_U_4 = vec(:,5);
press_4 = vec(:,15);

clear vec

vec = load('VEC01501509.dat');
vel_E_5= vec(:,3);
vel_N_5= vec(:,4);
vel_U_5 = vec(:,5);
press_5 = vec(:,15);

clear vec

vec = load('VEC01501510.dat');
vel_E_6= vec(:,3);
vel_N_6= vec(:,4);
vel_U_6 = vec(:,5);
press_6 = vec(:,15);

clear vec

vec = load('VEC01501511.dat');
vel_E_7= vec(:,3);
vel_N_7= vec(:,4);
vel_U_7 = vec(:,5);
press_7 = vec(:,15);

clear vec

vec = load('VEC01501512.dat');
vel_E_8= vec(:,3);
vel_N_8= vec(:,4);
vel_U_8 = vec(:,5);
press_8 = vec(:,15);

clear vec

vec = load('VEC01501513.dat');
vel_E_9= vec(:,3);
vel_N_9= vec(:,4);
vel_U_9 = vec(:,5);
press_9 = vec(:,15);

clear vec

vec = load('VEC01501514.dat');
vel_E_10= vec(:,3);
vel_N_10= vec(:,4);
vel_U_10 = vec(:,5);
press_10 = vec(:,15);

clear vec

vec = load('VEC01501515.dat');
vel_E_11= vec(:,3);
vel_N_11= vec(:,4);
vel_U_11 = vec(:,5);
press_11 = vec(:,15);

clear vec

vec = load('VEC01501516.dat');
vel_E_12= vec(:,3);
vel_N_12= vec(:,4);
vel_U_12 = vec(:,5);
press_12 = vec(:,15);

clear vec

vec = load('VEC01501517.dat');
vel_E_13= vec(:,3);
vel_N_13= vec(:,4);
vel_U_13 = vec(:,5);
press_13 = vec(:,15);

clear vec

vec = load('VEC01501518.dat');
vel_E_14= vec(:,3);
vel_N_14= vec(:,4);
vel_U_14 = vec(:,5);
press_14 = vec(:,15);

clear vec

vec = load('VEC01501519.dat');
vel_E_15= vec(:,3);
vel_N_15= vec(:,4);
vel_U_15 = vec(:,5);
press_15 = vec(:,15);

clear vec

vec = load('VEC01501520.dat');
vel_E_16= vec(:,3);
vel_N_16= vec(:,4);
vel_U_16 = vec(:,5);
press_16 = vec(:,15);

clear vec

vec = load('VEC01501521.dat');
vel_E_17= vec(:,3);
vel_N_17= vec(:,4);
vel_U_17 = vec(:,5);
press_17 = vec(:,15);

clear vec

vec = load('VEC01501522.dat');
vel_E_18= vec(:,3);
vel_N_18= vec(:,4);
vel_U_18 = vec(:,5);
press_18 = vec(:,15);

clear vec

vec = load('VEC01501523.dat');
vel_E_19= vec(:,3);
vel_N_19= vec(:,4);
vel_U_19 = vec(:,5);
press_19 = vec(:,15);

clear vec

vec = load('VEC01501524.dat');
vel_E_20= vec(:,3);
vel_N_20= vec(:,4);
vel_U_20 = vec(:,5);
press_20 = vec(:,15);

clear vec

vec = load('VEC01501525.dat');
vel_E_21= vec(:,3);
vel_N_21= vec(:,4);
vel_U_21 = vec(:,5);
press_21 = vec(:,15);

clear vec

vec = load('VEC01501526.dat');
vel_E_22= vec(:,3);
vel_N_22= vec(:,4);
vel_U_22 = vec(:,5);
press_22 = vec(:,15);

clear vec

vec = load('VEC01501527.dat');
vel_E_23= vec(:,3);
vel_N_23= vec(:,4);
vel_U_23 = vec(:,5);
press_23 = vec(:,15);

clear vec

vec = load('VEC01501528.dat');
vel_E_24= vec(:,3);
vel_N_24= vec(:,4);
vel_U_24 = vec(:,5);
press_24 = vec(:,15);

clear vec
%%  2  %%%%%%  Mean Day  %%%%%%%%  mean values per day   %%%%%%%%%%%%%%%%%%
x = vel_E_1;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_1 = valE';
 mean_E_1_date = val_E_1/32;
 
 y = vel_N_1;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_1 = valN';
 mean_N_1_date = val_N_1/32;
 
 z = vel_U_1;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_1 = valU';
 mean_U_1_date = val_U_1/32;
 %_______________________________________________________________ 
 
  x = vel_E_2;

for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_2 = valE';
 mean_E_2_date = val_E_2/32;
 
 y = vel_N_2;

for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_2 = valN';
  mean_N_2_date = val_N_2/32;
 
 z = vel_U_2;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_2 = valU';
 mean_U_2_date = val_U_2/32;
 %______________________________________________________________________ 
 
 x = vel_E_3;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_3 = valE';
 mean_E_3_date = val_E_3/32;
 
 y = vel_N_3;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_3 = valN';
 mean_N_3_date = val_N_3/32;
 
 z = vel_U_3;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_3 = valU';
 mean_U_3_date = val_U_3/32;
%______________________________________________________________________
 x = vel_E_4;

for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_4 = valE';
 mean_E_4_date = val_E_4/32;
 
 y = vel_N_4;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_4 = valN';
 mean_N_4_date = val_N_4/32;
 
 z = vel_U_4;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_4 = valU';
 mean_U_4_date = val_U_4/32;
%_________________________________________________________________________
 x = vel_E_5;
for  i = 1:((length(x)/32)-1) 
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_5 = valE';
 mean_E_5_date = val_E_5/32;
 
 y = vel_N_5;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_5 = valN';
  mean_N_5_date = val_N_5/32;
 
 z = vel_U_5;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_5 = valU';
 mean_U_5_date = val_U_5/32;
%_________________________________________________________________________ 

x = vel_E_6;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_6 = valE';
 mean_E_6_date = val_E_6/32;
 
 y = vel_N_6;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_6 = valN';
 mean_N_6_date = val_N_6/32;
 
 z = vel_U_6;

 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_6 = valU';
 mean_U_6_date = val_U_6/32;
%_________________________________________________________________________

x = vel_E_7;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_7 = valE';
 mean_E_7_date = val_E_7/32;
 
 y = vel_N_7;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_7 = valN';
 mean_N_7_date = val_N_7/32;
 
 z = vel_U_7;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_7 = valU';
 mean_U_7_date = val_U_7/32;
%_________________________________________________________________________
x = vel_E_8;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_8 = valE';
 mean_E_8_date = val_E_8/32;

 y = vel_N_8;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_8 = valN';
 mean_N_8_date = val_N_8/32;
 
 z = vel_U_8;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_8 = valU';
 mean_U_8_date = val_U_8/32;
%_________________________________________________________________________

x = vel_E_9;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_9 = valE';
 mean_E_9_date = val_E_9/32;

 y = vel_N_9;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_9 = valN';
 mean_N_9_date = val_N_9/32;
 
 z = vel_U_9;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_9 = valU';
 mean_U_9_date = val_U_9/32;
%_________________________________________________________________________ 
x = vel_E_10;

for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_10 = valE';
 mean_E_10_date = val_E_10/32;
 
 y = vel_N_10;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_10 = valN';
  mean_N_10_date = val_N_10/32;
 
 z = vel_U_10;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_10 = valU';
 mean_U_10_date = val_U_10/32;
%_________________________________________________________________________
x = vel_E_11;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_11 = valE';
 mean_E_11_date = val_E_11/32;
 
 y = vel_N_11;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_11 = valN';
 mean_N_11_date = val_N_11/32;
 
 z = vel_U_11;
  for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_11 = valU';
  mean_U_11_date = val_U_11/32;
%_________________________________________________________________________
x = vel_E_12;

for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_12 = valE';
 mean_E_12_date = val_E_12/32;
 
 y = vel_N_12;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_12 = valN';
 mean_N_12_date = val_N_12/32;
 
 z = vel_U_12; 
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_12 = valU';
 mean_U_12_date = val_U_12/32;
%_________________________________________________________________________

x = vel_E_13;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_13 = valE';
 mean_E_13_date = val_E_13/32;
 
 y = vel_N_13;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_13 = valN';
 mean_N_13_date = val_N_13/32;
 
 z = vel_U_13;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_13 = valU';
 mean_U_13_date = val_U_13/32;
%_________________________________________________________________________
x = vel_E_14;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_14 = valE';
 mean_E_14_date = val_E_14/32;
 
 y = vel_N_14;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_14 = valN';
 mean_N_14_date = val_N_14/32;
 
 z = vel_U_14;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_14 = valU';
 mean_U_14_date = val_U_14/32;
%_________________________________________________________________________ 
x = vel_E_15;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_15 = valE';
 mean_E_15_date = val_E_15/32;
 
 y = vel_N_15;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_15 = valN';
 mean_N_15_date = val_N_15/32;
 
 z = vel_U_15;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_15 = valU';
 mean_U_15_date = val_U_15/32;
%_________________________________________________________________________
 
x = vel_E_16;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_16 = valE';
 mean_E_16_date = val_E_16/32;
 
 y = vel_N_16;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_16 = valN';
 mean_N_16_date = val_N_16/32;
 
 z = vel_U_16;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_16 = valU';
 mean_U_16_date = val_U_16/32;
%_________________________________________________________________________
x = vel_E_17;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_17 = valE';
 mean_E_17_date = val_E_17/32;
 
 y = vel_N_17;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_17 = valN';
 mean_N_17_date = val_N_17/32;
 
 z = vel_U_17;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_17 = valU';
 mean_U_17_date = val_U_17/32;
%_________________________________________________________________________
x = vel_E_18;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_18 = valE';
 mean_E_18_date = val_E_18/32;
 
 y = vel_N_18;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_18 = valN';
 mean_N_18_date = val_N_18/32;
 
 
 z = vel_U_18;
for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_18 = valU';
 mean_U_18_date = val_U_18/32;
%_________________________________________________________________________
x = vel_E_19;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_19 = valE';
 mean_E_19_date = val_E_19/32;
 
 y = vel_N_19;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_19 = valN';
 mean_N_19_date = val_N_19/32;
 
 z = vel_U_19;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_19 = valU';
 mean_U_19_date = val_U_19/32;
%_________________________________________________________________________
x = vel_E_20;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_20 = valE';
 mean_E_20_date = val_E_20/32;
 
 y = vel_N_20;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_20 = valN';
 mean_N_20_date = val_N_20/32;
 
 z = vel_U_20;
 for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_20 = valU';
 mean_U_20_date = val_U_20/32;
%_________________________________________________________________________
x = vel_E_21;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_21 = valE';
 mean_E_21_date = val_E_21/32;
 
 y = vel_N_21;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_21 = valN';
 mean_N_21_date = val_N_21/32;
 
 z = vel_U_21;
for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_21 = valU';
 mean_U_21_date = val_U_21/32;
%_________________________________________________________________________
x = vel_E_22;

for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_22 = valE';
 mean_E_22_date = val_E_22/32;

 y = vel_N_22;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_22 = valN';
 mean_N_22_date = val_N_22/32;

 z = vel_U_22;
for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_22 = valU';
 mean_U_22_date = val_U_22/32;
%_________________________________________________________________________
x = vel_E_23;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_23 = valE';
 mean_E_23_date = val_E_23/32;
 
 y = vel_N_23;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_23 = valN';
 mean_N_23_date = val_N_23/32;
 
 z = vel_U_23;
for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_23 = valU';
 mean_U_23_date = val_U_23/32;
%_________________________________________________________________________
x = vel_E_24;

for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
valE(i) = sum(x(index(i):(index(i+1)-1)));
valE(600) = sum(x(19168:19200));
end
 val_E_24 = valE';
 mean_E_24_date = val_E_24/32;
 
 y = vel_N_24;
for  i = 1:((length(y)/32)-1)
index = 1:32:length(y);
valN(i) = sum(y(index(i):(index(i+1)-1)));
valN(600) = sum(y(19168:19200));
end
 val_N_24 = valN';
 mean_N_24_date = val_N_24/32;
 
 z = vel_U_24;
for  i = 1:((length(z)/32)-1)
index = 1:32:length(z);
valU(i) = sum(z(index(i):(index(i+1)-1)));
valU(600) = sum(z(19168:19200));
end
 val_U_24 = valU';
 mean_U_24_date = val_U_24/32;
 
%%  3  %%%%%%%%%%%%     Velocity    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

vec_E_date = [mean_E_1_date;mean_E_2_date;mean_E_3_date;mean_E_4_date;mean_E_5_date;mean_E_6_date;mean_E_7_date;...
    mean_E_8_date;mean_E_9_date;mean_E_10_date;mean_E_11_date;mean_E_12_date;mean_E_13_date;mean_E_14_date;mean_E_15_date;...
    mean_E_16_date;mean_E_17_date;mean_E_18_date;mean_E_19_date;mean_E_20_date;mean_E_21_date;...
    mean_E_22_date;mean_E_23_date;mean_E_24_date;];

vec_N_date = [mean_N_1_date;mean_N_2_date;mean_N_3_date;mean_N_4_date;mean_N_5_date;mean_N_6_date;mean_N_7_date;...
    mean_N_8_date;mean_N_9_date;mean_N_10_date;mean_N_11_date;mean_N_12_date;mean_N_13_date;mean_N_14_date;...
    mean_N_15_date;mean_N_16_date;mean_N_17_date;mean_N_18_date;mean_N_19_date;mean_N_20_date;mean_N_21_date;...
    mean_N_22_date;mean_N_23_date;mean_N_24_date;];

vec_U_date = [mean_U_1_date;mean_U_2_date;mean_U_3_date;mean_U_4_date;mean_U_5_date;mean_U_6_date;mean_U_7_date;...
    mean_U_8_date;mean_U_9_date;mean_U_10_date;mean_U_11_date;mean_U_12_date;mean_U_13_date;mean_U_14_date;...
    mean_U_15_date;mean_U_16_date;mean_U_17_date;mean_U_18_date;mean_U_19_date;mean_U_20_date;mean_U_21_date;...
    mean_U_22_date;mean_U_23_date;mean_U_24_date;];

figure1 = figure;
subplot(3,1,1)
plot(vec_E_date,'r')
axis([0 14400 -1.0 1.0])
title('Vector - Velocity East - date')

subplot(3,1,2)
plot(vec_N_date,'b')
axis([0 14400 -1.0 1.0])
title('Vector - Velocity North - date')

subplot(3,1,3)
plot(vec_U_date,'m')
axis([0 14400 -1.0 1.0])
title('Vector - Velocity Up - date')

%% 4 %%%%%%%%%%      pressure data       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = press_1;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_1 = press_mat';
 mean_press_1_date = val_press_1/32;
 
  x = press_2;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_2 = press_mat';
  mean_press_2_date = val_press_2/32;
 
 x = press_3;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_3 = press_mat';
 mean_press_3_date = val_press_3/32;
 
 x = press_4;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_4 = press_mat';
 mean_press_4_date = val_press_4/32;
 
  x = press_5;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_5 = press_mat';
 mean_press_5_date = val_press_5/32;
 
 x = press_6;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_6 = press_mat';
 mean_press_6_date = val_press_6/32;
 
 x = press_7;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_7 = press_mat';
 mean_press_7_date = val_press_7/32;
 
 x = press_8;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_8 = press_mat';
 mean_press_8_date = val_press_8/32;
 
 x = press_9;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_9 = press_mat';
 mean_press_9_date = val_press_9/32;
 
 x = press_10;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_10 = press_mat';
 mean_press_10_date = val_press_10/32;
 
 x = press_11;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_11 = press_mat';
 mean_press_11_date = val_press_11/32;
 
 x = press_12;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_12 = press_mat';
 mean_press_12_date = val_press_12/32;
 
 x = press_13;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_13 = press_mat';
 mean_press_13_date = val_press_13/32;
 
 x = press_14;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_14 = press_mat';
 mean_press_14_date = val_press_14/32;
 
 x = press_15;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_15 = press_mat';
 mean_press_15_date = val_press_15/32;
 
 x = press_16;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_16 = press_mat';
 mean_press_16_date = val_press_16/32;
 
 x = press_17;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_17 = press_mat';
 mean_press_17_date = val_press_17/32;
 
 x = press_18;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_18 = press_mat';
 mean_press_18_date = val_press_18/32;
 
 x = press_19;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_19 = press_mat';
 mean_press_19_date = val_press_19/32;
 
 x = press_20;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_20 = press_mat';
 mean_press_20_date = val_press_20/32;
 
 x = press_21;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_21 = press_mat';
 mean_press_21_date = val_press_21/32;
 
 x = press_22;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_22 = press_mat';
 mean_press_22_date = val_press_22/32;
 
 x = press_23;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_23 = press_mat';
 mean_press_23_date = val_press_23/32;
 
 x = press_24;
for  i = 1:((length(x)/32)-1)
index = 1:32:length(x);
press_mat(i) = sum(x(index(i):(index(i+1)-1)));
press_mat(600) = sum(x(19168:19200));
end
 val_press_24 = press_mat';
  mean_press_24_date = val_press_24/32;
 
press_date = [mean_press_1_date;mean_press_2_date;mean_press_3_date;mean_press_4_date;mean_press_5_date;mean_press_6_date;mean_press_7_date;...
    mean_press_8_date;mean_press_9_date;mean_press_10_date;mean_press_11_date;mean_press_12_date;mean_press_13_date;mean_press_14_date;mean_press_15_date;...
    mean_press_16_date;mean_press_17_date;mean_press_18_date;mean_press_19_date;mean_press_20_date;mean_press_21_date;...
    mean_press_22_date;mean_press_23_date;mean_press_24_date;];

surface_vec = ((press_date)*1.019716) + 0.56 + 0.217;

figure2 = figure

subplot(2,1,1)
plot(press_date,'r')
axis([0 14400 0 15])
title('Pressure')

subplot(2,1,2)
plot(surface_vec,'b')
axis([0 14400 0 15])
title('Surface - vec')

%% 5 %%%%%%%%%%%   turb vec    %%%%%%%%%%%%  needs to check %%%%%%%%%%%%%%%

%vel_date = sqrt(((vec_E_date).^2)+((vec_N_date).^2));

vel3_date = sqrt(((vec_E_date).^2)+((vec_N_date).^2)+((vec_U_date).^2));

%TKE_date = 0.5*(vel3_date);        %Total Kinatic Energy??     % from velocity components

ut_date = vec_E_date - mean (vec_E_date);     % u
vt_date = vec_N_date - mean (vec_N_date);     % v
wt_date = vec_U_date - mean (vec_U_date);     % w
 
turb_TKE_date = 0.5*((ut_date.^2) + (vt_date.^2) + (wt_date.^2));

Reynolds_Stress = ut_date .* wt_date;

R_st_date = Reynolds_Stress;

%% 6 %%%%%%%%%%%%%  direction  %%%%%%%   velocity vector   %%%%%%%%%%%%%%%%
 x = vec_E_date;
 y = vec_N_date;
% vel_date = sqrt(((vec_E_date).^2)+((vec_N_date).^2));
% vel = vel_date;
 
sign_x = sign(vec_E_date);
sign_y = sign(vec_N_date);

for i = 1:14400
alfa(i) = atan(y(i)./x(i));
if sign_x(i)>0 & sign_y(i)>0
alfa_rad(i) = alfa(i);
elseif sign_x(i)<0 & sign_y(i)>0 
    alfa_rad(i) = alfa(i) + pi;
elseif sign_x(i)<0 & sign_y(i)<0 
    alfa_rad(i) = alfa(i) + pi;
else
    alfa_rad(i) = alfa(i) + 2*pi;
end

alfa_dir_date(i) = 180*(alfa_rad(i))/pi;
end

figure3 = figure 
plot(alfa_dir_date)
axis([0 14400 0 360])
title('Velocitiy Direction')

%% 7 %%%%%%%%%%%%%%%%     AWAC  load   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wave = load('Herculesa660602_23092015_14112015.wap');

date_campaign = wave(:,1:6);
signif_height = wave(:,8);
wave_H3 = wave(:,9);
wave_max = wave(:,11);

wave_mean = wave(:,12);

%mean_period = wave(:,16);           %%% which one is correct
mean_period = wave(:,13);            %%% which one is correct
period_T3 = wave(:,16);
peak_period = wave(:,14);
max_period_Tmax = wave(:,18);
mean_direction = wave(:,21);
mean_pressure = wave(:,23);
current_speed = wave(:,29);
current_direction = wave(:,30);

sea_level = 1.40+((mean_pressure)*1.019716);

%% 8 %%%%%%%%%    Sea Level by AWAC vs Vec     %%%%%%%%%%%%%%%%%%%%%%%%%%%%
day = find((wave(:,1)==10&(wave(:,2)==14)));   % month 10 day 14

lev = sea_level(day);
t1 = [1:1200:14400];
see = zeros(14400,1);

for i = 1:14400
    see(i) = NaN;
end

see(t1) = lev;

figure4 = figure
t= [0:14400];
plot(surface_vec)
axis([0 14400 0 15])
title('Sea Level AWAC vs Vec')

hold on
plot(see, 'r:+')

%%  9   %%%%%%%%%%%%%%%%%    Total plot    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure5 = figure

t = [1:14400];

subplot(5,2,5)
plot(vec_E_date)                    % u
axis([0 14400 -0.5 0.5])
ylabel('u (m/s)')

subplot(5,2,7)
plot(vec_N_date)                    % v
axis([0 14400 -0.5 0.5])
ylabel('v (m/s)')

subplot(5,2,9)
plot(vec_U_date)                    % w
axis([0 14400 -0.5 0.5])
ylabel('w (m/s)')

subplot(5,2,1)
plot(vel3_date)
axis([0 14400 0 0.8])
%ylabel('h vel (m/s)')
ylabel('vel (m/s)')

subplot(5,2,3)
plot(alfa_dir_date)
axis([0 14400 0 360])
ylabel('vel dir (\circ)')

subplot(5,2,4)
plot(turb_TKE_date)
axis([0 14400 0 0.3])
ylabel('TKE (m^2/s^2)')

subplot(5,2,8)
%plot(t,mean1_date,t,mean2_date)
axis([0 14400 0 40])
ylabel('SSC (mg/l)')

subplot(5,2,2)
plot(surface_vec)
axis([0 14400 0 14])
hold on
plot(see, 'r:+')
ylabel('depth (m)')

subplot(5,2,10)
%plot(mean3_date)
axis([0 14400 0 200])
ylabel('SSC (mg/l)')

subplot(5,2,6)
plot(R_st_date)
axis([0 14400 -0.1 0.1])
ylabel('u''w'' (m^2/s^2)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
