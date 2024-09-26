# -*- coding: utf-8 -*-
"""
#******************************************************************************
# This program computes 15 min wave spectra 
#
#  Author: Dieter Vanneste
#  Dec 2022
#******************************************************************************
"""
import os
import sys
from pylab import * 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy import signal
#from scipy.signal import find_peaks, butter, peak_prominences, lfilter
import scipy.fftpack
import math


def calculate_Sf_params(S,f,fmin,fmax,delta_f):  

    if np.isnan(S).all():
        Hm0,Tm01,Tm02,Tm_10 = np.nan,np.nan,np.nan,np.nan
    
    else:
        Sf_pd = pd.DataFrame({'f':f,'Sf':S})
        fp = f[Sf_pd['Sf'][(Sf_pd['f']>=fmin) &(Sf_pd['f']<=fmax)].idxmax()]  #search fp between fmin and fmax
                   
        m0,m1,m2,m_1 = 0,0,0,0
        index_f = [i[0] for i in np.argwhere((f>=min(fp/3,fmin)) & (f<=min(3*fp,fmax)))] 
        for j in range(index_f[0],index_f[-1]+1):
            m0 = m0 + S[j]*delta_f  #significant wave height is computed in frequency interval min(fp/3;0.05Hz)-min(3*fp;0.43Hz), see $3.5 IMDC report RA15039
            m1 = m1 + f[j]*S[j]*delta_f
            m2 = m1 + (f[j]**2)*S[j]*delta_f
            m_1 = m_1 + (f[j]**(-1))*S[j]*delta_f
        Hm0 = 4*np.power(m0,0.5)
        Tm01 = m0/m1
        Tm02 = m0/m2
        Tm_10 = m_1/m0

    return Hm0,Tm01,Tm02,Tm_10

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

location = 'Jachthaven'
output_suffix=''
fs = 8 #sample rate eta file [Hz]
process_psensor = True
process_radac = not process_psensor

start_time ='2022-01-01 00:00:00' 
end_time='2022-02-01 00:00:00'
st = pd.to_datetime(start_time)
te = pd.to_datetime(end_time)
titlest = st.strftime('%Y-%m-%d')

#selected window to export Sf data files
st_Sf_out =pd.to_datetime('2022-01-31 08:00:00')
et_Sf_out=pd.to_datetime('2022-01-31 16:00:00')

#basedir = os.path.join('P:\\','20_016-CPnsZGolfklim','3_Uitvoering','Python','wave analysis')
#basedir = r'C:\\Users\vannesdi\\Documents\\work files\\20_025\\stormanalyse_laagtij\\storm_10102013'
basedir = r'C:\Users\vannesdi\Documents\work files\18_053\storm Corrie'

#Spectrum_folder =os.path.join(basedir,location+'_Shortwaves'+st.strftime('%Y_%m_%d')+'_'+te.strftime('%Y_%m_%d'))


#datafile_eta = os.path.join(basedir,'eta','eta_'+location+'_'+year+'_'+month+'_'+day_start+'_'+year+'_'+month+'_'+day_end+'.dat')
if process_psensor:
    datafile_eta = os.path.join(basedir,'druksensoren','eta',location+'_eta_'+'%i' %fs+'Hz_'+st.strftime('%Y_%m_%d')+'_'+te.strftime('%Y_%m_%d')+output_suffix+ '.dat')
    Spectrum_folder =os.path.join(basedir,'druksensoren','wave spectra 15 min','%i' %fs +'Hz')
    Spectrum_folder =os.path.join(basedir,'druksensoren','wave spectra 15 min_fs'+'%i' %fs +'Hz_M1024')
elif process_radac:
    datafile_eta = os.path.join(basedir,'radac_'+location,'eta',location+'_eta_'+'%i' %fs +'Hz_'+st.strftime('%Y_%m_%d')+'_'+te.strftime('%Y_%m_%d')+output_suffix+ '.dat')
    Spectrum_folder =os.path.join(basedir,'radac_'+location,'wave spectra 15 min_fs'+'%i' %fs +'Hz')
    Spectrum_folder =os.path.join(basedir,'radac_'+location,'wave spectra 15 min_fs'+'%i' %fs +'Hz_M1024')

if not os.path.exists(Spectrum_folder):
    os.makedirs(Spectrum_folder)

fileT = open(datafile_eta, 'r')
eta = pd.read_csv(datafile_eta,header=None)
eta_np=eta.iloc[:,0:].values.flatten()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# parameters in spectral analysis
m = 15*60*fs                            #number of data points in analysis window used for computation of spectral parameters (15-min in RA15039)
M = 128*fs                              #segment length for spectral averaging of full spectrum based on 15 min windows (Table 3-1 IMDC report RA15039) 
p = 12                                   #number of segments (Table 3-2 IMDC report RA15039) 
delta_f = fs/M                          #spectral resolution
Noverlap = math.floor((m-p*M)/(1-p))    #length of overlap between segments (adjusted to length of window)

f = np.array([i*delta_f for i in range(int(M/2))])  
f_full = np.array([i*delta_f for i in range(M)])  
f = f.T

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#move window (with m data points) over elevation time series

Neta = len(eta_np)
Nwindows = Neta//m
#Hm0,fp,Tp,Tm01,Tm02,Tm_10 = np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows) #initialize arrays of spectral parameters
#Hm0cos,fpcos,Tpcos,Tm01cos,Tm02cos,Tm_10cos = np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows),np.zeros(Nwindows) #initialize arrays of spectral parameters

#define taper
Ntaper=int(0.1*M)   #taper width specified in IMDC report RA15039 
cos_taper = np.array([0.5*(1-np.cos(math.pi*i/Ntaper)) for i in range(0,Ntaper)]+
                     [1.0 for i in range(Ntaper,M-Ntaper)]+[0.5*(1+np.cos(math.pi*(i-(M-Ntaper))/Ntaper)) for i in range(M-Ntaper,M)])   
cos_taper = cos_taper.T

#loop over windows
for i in range(0,Nwindows):
    eta_np_window = eta_np[i*m:(i+1)*m]
    notnan = np.argwhere(~np.isnan(eta_np_window))
    eta_np_window_notnan = eta_np_window[notnan].flatten() #remove NaN values in segment (for fft)
    m_star = len(eta_np_window_notnan)        
    p_check = p
    count = 0
    
    if m_star == 0: #check if window contains any non NaN record
        S,Scos = np.full(int(M/2),np.nan),np.full(int(M/2),np.nan)
    elif m_star > 0: 
        if m_star < m: 
            p_star = math.ceil((m_star-Noverlap)/(M-Noverlap))
            p_check = p_star   
        
        N = p_check*M-(p_check-1)*Noverlap    #length of array zero-padded to nearest multiple of M, spanning p segments with Noverlap
        S,Scos = np.zeros(int(M/2)),np.zeros(int(M/2))
        
        #loop over segments
        for q in range(0,N-(M-Noverlap),M-Noverlap): 
            o = min(q + M, m_star)
            etaseg = eta_np_window_notnan[q:o]
            seg_len=len(etaseg)
            trend = etaseg-signal.detrend(etaseg) #remove linear trend  
            etaseg = etaseg-trend
            
            if seg_len<M:  # zero-pads to nearest length M
                result = np.zeros(M)
                result[:seg_len] = etaseg
                etaseg = result		 
            
            S_eta = np.fft.fft(np.multiply(etaseg,np.hanning(M)))  
            A_k = np.real(S_eta)/M           #scale factor not included in numpy fft routine, see (C.43) appendix C PhD Peter Troch
            B_k = np.imag(S_eta)/M
            S_k = 2*(np.power(A_k,2)+np.power(B_k,2))/delta_f #raw spectrum,  eq. (C.48) App. C Troch
    #        S_k = S_k*(8/3)      #compensation for energy loss due to Hanning window ($3.4 IMDC report RA15039) 
            S_k = S_k*(M/sum(np.hanning(M)**2)) #compensation for energy loss due to Hanning window, , see (C.51) appendix C PhD Peter Troch
            S = np.add(S,S_k[:int(M/2)]) 
            
            #compute spectrum applying cos taper
            S_eta = np.fft.fft(np.multiply(etaseg,cos_taper))
            A_k = np.real(S_eta)/M           #scale factor not included in numpy fft routine, see (C.43) appendix C PhD Peter Troch
            B_k = np.imag(S_eta)/M
            S_k = 2*(np.power(A_k,2)+np.power(B_k,2))/delta_f #raw spectrum,  eq. (C.48) App. C Troch
            S_k = S_k*(M/sum(cos_taper**2)) #compensation for energy loss due to Hanning window, , see (C.51) appendix C PhD Peter Troch
            Scos = np.add(Scos,S_k[:int(M/2)]) 
            
            count+=1
                
        if count==p_check:
            S = S/p_check   #averaged spectrum in time window M
            Scos = Scos/p_check
        else:
            ind_start,ind_end = i*m,(i+1)*m
            print('i=' +'%i' %i +' q='+ '%i' %q)
            print('window n='+ '%i' %ind_start +'..'+'%i' %ind_end)        
            print('error in calculation of averaged spectrum, please check')    
       
#        fp[i]=f[S.argmax()] 
#        Tp[i] = 1/fp[i]
#        fpcos[i]=f[Scos.argmax()]
        
#    Hm0[i],Tm01[i],Tm02[i],Tm_10[i] = calculate_Sf_params(S,f,0.05,0.43,delta_f)
#    Hm0cos[i],Tm01cos[i],Tm02cos[i],Tm_10cos[i] = calculate_Sf_params(Scos,f,0.05,0.43,delta_f)
    
     #output spectrum files in specified time window
    time = pd.to_datetime(start_time)+i*pd.Timedelta('15 min')
    if st_Sf_out <= time <= et_Sf_out:
        print('output Sf at '+ datetime.strftime(time,'%d-%m-%Y %H:%M'))
        data_Sf = pd.DataFrame({'f':f,'Sf':S})        
        data_Sf.to_csv(os.path.join(Spectrum_folder,'Sf_'+location+'_'+datetime.strftime(time,'%d-%m-%Y %H-%M')+'.txt'),sep='\t',header=True,index=None)
         
#H_times = pd.date_range(pd.to_datetime(start_time)+pd.Timedelta('15 min'),pd.to_datetime(start_time)+(len(Hm0))*pd.Timedelta('15 min'),freq='15 min')
#spectrum_out=pd.DataFrame({'Hm0 [m]':Hm0,'Tm-10 [s]':Tm_10},index=H_times)
#
#plt.plot(spectrum_out['Hm0 [m]'],'ko-')
#spectrum_out.to_csv(os.path.join(Spectrum_folder,'Hm0_Tm_10_'+location+'_15min.txt'),header=True,sep='\t',float_format = '%.3e')
#
#spectrum_cos_out=pd.DataFrame({'Hm0':Hm0cos},index=H_times)
#plt.plot(spectrum_cos_out['Hm0'],'bs:')

#Hm0_IMDC = pd.read_csv(os.path.join(basedir,'imdc','20131001000000_20131031234500_value_Hm0_SW_15M.txt'),sep='\s+',index_col=False,header=2,names=['tijd','NPT-A','NPT-B','NPT-C','NPT-D','BLK-A','BLK-B','BLK-C','ZBG-A','ZBG-B','ZBG-C','OST-A','OST-B','OST-C','OST-D','OST-E'])
#Hm0_IMDC = Hm0_IMDC.replace(-9,np.nan)
#Hm0_IMDC_times = pd.date_range(pd.to_datetime('2013-10-01 00:00:00'),pd.to_datetime('2013-10-31 23:45:00'),freq='15 min')
#plt.plot(Hm0_IMDC_times,Hm0_IMDC['OST-A'],'r+')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  plots

#def cm2inch(*tupl):
#        inch = 2.54
#        if isinstance(tupl[0], tuple):
#            return tuple(i/inch for i in tupl[0])
#        else:
#            return tuple(i/inch for i in tupl)    
#
#fig, ax1 = plt.subplots(figsize=cm2inch(45,22),sharey=True)  
#
#title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal',
#              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#axis_font = {'fontname':'Arial', 'size':'20'}
#
#plt.title('Storm event '+titlest+' - '+location+': short waves',**title_font)
#ax1.set_ylabel('$H_{m0}$ [m], $η$ [m]', fontsize='18',labelpad=20)  
#lns1 = ax1.plot(spectrum_out['Hm0'], 'ro',label='$H_{m0}$', markersize=4,zorder=2)
#H_times_eta = pd.date_range(pd.to_datetime(start_time),pd.to_datetime(start_time)+(len(eta_np)-1)*pd.Timedelta(1/fs,'s'),freq='%i' %int(1000/fs) +'L')
#lns2 = ax1.plot(H_times_eta,eta_np,'C7-', label='$η_0$', linewidth=0.5, zorder=1)
##ax1.tick_params(axis='Hm0', label='Hm0 (m), η0 (m)', fontsize='14')
#lns3 = ax1.plot(Hm0_IMDC_times,Hm0_IMDC['OST-A'],'ks',label='$H_{m0} (IMDC)$', markersize=5,zorder=2,fillstyle='none')
#
#plt.xlim((pd.to_datetime(start_time),pd.to_datetime(end_time)))
#plt.ylim((-0.7,0.7))
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#
#plt.yticks(fontsize=14)
#plt.xticks(fontsize=14)
#
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#ax2.set_xlabel('time',fontsize='18',labelpad=20)
#ax2.set_ylabel('tide [m TAW]', fontsize='18',labelpad=20)
#lns4 = ax2.plot(data_tide['time'].values,data_tide['mTAW'].values,'k-',label='water level',zorder=10)
#ax2.tick_params(axis='y')
#plt.ylim((0,6.0))
#
#plt.yticks(fontsize=14)
#
## Solution for having two legends
#lns = lns1+lns2+lns3+lns4
#labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc='upper right',fontsize='18')
#
#for ax in [ax1]:
##    ax.xaxis.grid(True, which='major') # `which` can be 'minor', 'major', or 'both'
#    ax.yaxis.grid(True, which='major',linestyle='--')
#fig.tight_layout()
#
#fig1_file=os.path.join(Spectrum_folder,location+'_SW_'+st.strftime('%Y_%m_%d')+'_'+te.strftime('%Y_%m_%d')+ '.png')
#fig.savefig(fig1_file)   # save the figure to file
#plt.close(fig) 
#
#
#
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#fig, ax1 = plt.subplots(figsize=cm2inch(40,60),sharey=True)
#
#title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal',
#              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#axis_font = {'fontname':'Arial', 'size':'16'}
#
#
#
#plt.subplot(5, 1, 1)
#plt.plot(Time,Hm0,'r*')
##plt.xlabel('time',**axis_font)
#plt.ylabel('Hm0 (m)',**axis_font)
#plt.ylim((0,0.4))
#
#plt.xticks(fontsize=11)
#plt.yticks(fontsize=14)
#plt.grid(axis='y', linestyle='--')
#plt.title('Short wave characteristics - '+location+' - '+titlest,**title_font)
#
#plt.subplot(5, 1, 2)
#plt.plot(Time,Tp,'b-')
#plt.ylabel('Tp (s)',**axis_font)
#plt.ylim((2,10))
#plt.xticks(fontsize=11)
#plt.yticks(fontsize=14)
#plt.grid(axis='y', linestyle='--')
#
#plt.subplot(5, 1, 3)
#plt.plot(Time, Tm01,'b-')
#plt.ylabel('Tm01 (s)',**axis_font)
#plt.ylim((2,10))
#ax.grid()
#plt.xticks(fontsize=11)
#plt.yticks(fontsize=14)
#plt.grid(axis='y', linestyle='--')
#
#plt.subplot(5, 1, 4)
#plt.plot(Time, Tm02,'b-')
#plt.ylabel('Tm02 (s)',**axis_font)
#plt.ylim((2,10))
#
#plt.xticks(fontsize=11)
#plt.yticks(fontsize=14)
#plt.grid(axis='y', linestyle='--')
#
#plt.subplot(5, 1, 5)
#plt.plot(Time, Tm_10,'b-')
#plt.ylabel('Tm-10 (s)',**axis_font)
#plt.ylim((2,10))
#
#plt.xticks(fontsize=11)
#plt.yticks(fontsize=14)
#plt.grid(axis='y', linestyle='--')
#fig.tight_layout()
#
#fig2_file=os.path.join(basedir,location+'_HT_SW'+st.strftime('%Y_%m_%d')+'_'+te.strftime('%Y_%m_%d')+ '.png')
#fig.savefig(fig2_file)   # save the figure to file
#plt.close(fig) 

