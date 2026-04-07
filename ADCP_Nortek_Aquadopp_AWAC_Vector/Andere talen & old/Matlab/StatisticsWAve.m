clear
clc

temp=load('Herculesa660602_23092015_14112015.wap');

Hm0=temp(:,8);
Hmax=temp(:,11);
Tp=temp(:,14);
Mdir=temp(:,21);

%stat for Hm0
AvgHm0=mean(Hm0);
SDHm0=std(Hm0);
kurtHm0=kurtosis(Hm0);
skewnessHm0=skewness(Hm0);
MaxHm0=max(Hm0);
MinHm0=min(Hm0);

break
%stat for Hmax
AvgHmax=mean(Hmax);
SDHmax=std(Hmax);
kurtHmax=kurtosis(Hmax);
skewnessHmax=skewness(Hmax);
MaxHmax=max(Hmax);
MinHmax=min(Hmax);
break
%stat for Tp
AvgTp=mean(Tp);
SDTp=std(Tp);
kurtTp=kurtosis(Tp);
skewnessTp=skewness(Tp);
MaxTp=max(Tp);
MinTp=min(Tp);

break
%stat for Mdir
AvgMdir=mean(Mdir);
SDMdir=std(Mdir);
kurtMdir=kurtosis(Mdir);
skewnessMdir=skewness(Mdir);
MaxMdir=max(Mdir);
MinMdir=min(Mdir);