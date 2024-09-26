#******************************************************************************
#This script converts a recorded pressure time series to elevation (eta) time series:

# (i)  complete eta time series including tidal wave
# (ii) short wave eta time series excluding tidal wave (f>=0.05 Hz)
# (iii) long wave eta time series including tidal wave (f<0.05 Hz)

# Dieter Vanneste, Jan 2022
#******************************************************************************
import os
import sys
from pylab import * 
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.fft import fft, fftshift
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, butter, peak_prominences, lfilter
import scipy.fftpack
import math

import subprocess

subprocess.run([
    r"C:\Users\claeysst\AppData\Local\Microsoft\WindowsApps\python3.12.exe",
    r"c:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\p_to_eta.py"
])


script_dir = os.path.dirname(__file__)
datafile_pt = os.path.join(script_dir, 'relative_path_to_data_file.dat')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function to calculate wave number based on water depth (m) and frequency f (hz)
def wavenum (depth, f):
    sigma=2*math.pi*f
    y=((np.square(sigma))*depth)/g
    y=np.transpose(y)
    k=np.zeros(len(f))   
    for i in range(0,len(f)):
        x=y[i]   
        for wn in range(1, 100):
            H=math.tanh(x)
            F=y[i]-(x*H)
            if abs(F)<0.000001:
                k[i]=x/depth
            else:
                FD=-H-x/math.sqrt(math.cosh(x))
                x=x-F/FD
                k[i]=x/depth
    return k

def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)  

def plot_p(p,window,ylim):
    
    custom_cycler = (cycler(color=['k','r', 'b', 'y', 'g'])*cycler(linestyle=['-', '--', ':']))
    if window == 'all':
        ts = pd.to_datetime('2022-01-01 00:00')
        te = pd.to_datetime('2022-02-01 00:00')
    else:
        ts,te = window[0],window[1]
    
    fig, ax = plt.subplots(figsize=cm2inch(20,15))
    plt.rc('axes', prop_cycle=custom_cycler)
    labelticksize='8'
    axtitlesize='10'
    legendfontsize='7'
    
    for pres in p:
        ax.plot(pres[0][ts:te],label=pres[1])

    ax.set_xlabel('time [UTC]',fontsize=axtitlesize,labelpad=10)
    ax.set_ylabel('pressure [Pa]',fontsize=axtitlesize,labelpad=10)
    
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
      
    ax.tick_params( direction='out', length=4, width=1, labelsize=labelticksize)    
    ax.grid(True, which='major', color='grey', linestyle=':')
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,0.1), bbox_transform=ax.transAxes,fontsize=legendfontsize)
    fig.tight_layout()
#    plt.savefig(os.path.join(basedir,'P(Pa)_'+p[0][1]+'_'+window[0].strftime('%m-%d %H:%M').replace(':','-')+'_'+window[1].strftime('%m-%d %H:%M').replace(':','-')+'.png'),dpi=300)
#    plt.close(fig)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#inputs
#basedir = 'C:\\Users\\vannesdi\\Documents\\work files\\18_053\\comp_radac_OSB\\'
basedir = r'C:\Users\vannesdi\Documents\work files\18_053\storm Corrie\druksensoren'
location = 'Jachthaven'
pdata_window=[2022,1]
year,month = pdata_window[0],pdata_window[1]

#constants and input:
g=9.81                      # gravity acceleration  
fs=8                        # sample freq [Hz]
write_output = False        #(de)activate output generation   
#output_suffix='_00-24'
output_suffix=''

# Selected time window
start_time ='2022-01-01 00:00:00' 
end_time='2022-02-01 00:00:00'

titlest = pd.to_datetime(start_time).strftime('%Y-%m-%d')

# sea water density [kg/m³],sensor elevation [m TAW] based on fit, elevation sensor [m] above bed 
info_psensor = [['Jachthaven',1026.18,-3.01,-4.2]]   # assumed bed level   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# pressure sensor and bed elevation
for i in range(0,len(info_psensor)):
    if info_psensor[i][0] == location:
        rho = info_psensor[i][1] #sea water density [kg/m³]       
        zbot = info_psensor[i][3]  #bed level [m TAW]
        zsens = info_psensor[i][2] #sensor elevation [m TAW]

#reading pressure signal
#datafile_pt =os.path.join(basedir,'Pseries_input',location+'_'+year+'_'+month+'_01_'+year+'_'+str(int(month)+1)+'_01.dat')
datafile_pt =os.path.join(basedir,'Pseries_input',location+'_'+str(year)+'_'+str(month).zfill(2)+'_01_'+str(year)+'_'+str(int(month)+1).zfill(2)+'_01_'+'%i' %fs+'Hz_clean.dat')
pt = pd.read_csv(datafile_pt, sep='\t')
pt['time']= pd.to_datetime(pt['time'])
pt = pt[(pt['time'] >= pd.to_datetime(start_time)) & (pt['time'] < pd.to_datetime(end_time))]
time=pt.iloc[:,0].values

pt_np=pt.iloc[:,1:].values
pt_np=pt_np.flatten()

#initialize
H_withnan,H_SW_withnan,H_LW_withnan = pt_np.copy(),pt_np.copy(),pt_np.copy()
notnan = np.argwhere(~np.isnan(pt_np))
pt_notnan = pt_np[notnan].flatten()

m = len(pt_notnan)
window = 15*60*fs             #division in time windows of 15 min (specified in IMDC report 15039)
M=int(window+1148*fs)         #addition of approx. 10 min to start and end of window (specified in IMDC report RA15039) 
Noverlap = M-window           #length of overlap between segments

p = math.ceil((m-Noverlap)/(M-Noverlap)) #estimate number of segments based on segment and overlap width
N = p*M-(p-1)*Noverlap        # length of array zero-padded to nearest multiple of M, spanning p segments with Noverlap

f = np.array([i*fs/M for i in range(int(M/2)+1)])  
f_full = np.array([i*fs/M for i in range(M+1)])  
f = f.T
H = np.empty(N)               #pre-allocate eta vector (unfiltered)
H_SW = np.empty(N)            #pre-allocate eta vector for short waves
H_LW = np.empty(N)            #pre-allocate eta vector for long waves

min_frequency = 0.05;		  #mininum frequency, below which no correction is applied (0.05 Hz)
max_frequency = 0.33;		  #maximum frequency, above which no correction is applied (0.33 Hz)

# dividing the pressure time series into segments

for q in range(0,N-(M-Noverlap),M-Noverlap): 
    o = min(q + M, m)
    ptseg = pt_notnan[q:o]
    seg_len=len(ptseg)
    trend = ptseg-signal.detrend(ptseg) #remove linear trend  
    ptseg = ptseg-trend
    
#    t=np.array([i for i in range(q,o)])
#==============================================================================    
    #compute wavenumber
    trend = trend/(g*rho) #convert to water column height
    h = trend.mean()  #water column height averaged in window
    depth = h+abs(zsens-zbot) #averaged total water depth in window
    k = np.array(wavenum (depth, f)) 
    k = k.T
#==============================================================================
    #transfer function (Kpt), Kpt<=25, between f=0.33 and 0.43 Hz gradually tending to zero by Hanning window
    Kpt = np.array(rho*g*np.cosh(k*abs(zsens-zbot))/np.cosh(k*depth))
    Kpt = Kpt.T
    TF = np.power(1/Kpt,1)
    N1 = 2*len(f[np.where((f>max_frequency) & (f<=0.43))])        
    han_win = np.hanning(N1)  #Hanning window taper
    
    #construct tapered TF limited to 0.43 Hz
    j=0
    for i in range(0,len(f)):
        TF[i] = min(TF[i],25) #limit TF to 25, see IMDC RA10037
        if f[i]>max_frequency and f[i]<=0.43:
            TF[i]=TF[i]*(1-han_win[j]) # apply Hanning window 
            j=j+1
        elif f[i]>0.43:
            TF[i]=0.0  # suppress all wave energy above 0.43 Hz

    TF = [TF[i] for i in range(0,int(M/2)+1)]+[TF[int(M/2)-i] for i in range(int(M/2)+1,M)]    # the second half is symmetrical
    
    #construct cosinus taper on both sides of window
    Ntaper=int(0.1*M)   #taper width specified in IMDC report RA15039 
    overlap_window = np.array([0.5*(1-np.cos(math.pi*i/Ntaper)) for i in range(0,Ntaper)]+
                              [1.0 for i in range(Ntaper,M-Ntaper)]+[0.5*(1+np.cos(math.pi*(i-(M-Ntaper))/Ntaper)) for i in range(M-Ntaper,M)])   
    overlap_window = overlap_window.T
 
#==============================================================================       
    if seg_len<M:  # zero-pads to length M
        result = np.zeros(M)
        result[:seg_len] = ptseg
        ptseg = result		 
        
    if q>0 and q<N-M: #apply taper
        ptseg2 = np.multiply(ptseg,overlap_window)
    else:
        ptseg2 = ptseg  #do not apply on first and last segment
    
    Ap=1   #no compensation needed since taper outside of retained window 
#    Ap=(np.var(ptseg)/np.var(ptseg2))**0.5  #compensation for energy loss due to taper window
    ptseg2 = ptseg2*Ap            
    S_p = np.fft.fft(ptseg2)      #pressure spectrum   
    S_eta = np.multiply(S_p,TF)   #elevation spectrum
    Hseg = np.real(np.fft.ifft(S_eta))
    Hseg = Hseg[:seg_len]
    
    #high-pass filter at 0.05 Hz for short waves elevation time series
    index_SW = [i[0] for i in np.argwhere((f_full>=0.05) & (f_full<=fs-0.05))] 
    S_eta_SW = S_eta.copy()
    S_eta_SW[0:index_SW[0]]=0
    S_eta_SW[index_SW[-1]+1:]=0
    Hseg_SW = np.real(np.fft.ifft(S_eta_SW))     #inverted FFT eta time series    
    Hseg_SW = Hseg_SW[:seg_len]
    
    #low-pass filter at 0.05 Hz for long waves elevation time series
    S_eta_LW = S_eta.copy()
    S_eta_LW[index_SW[0]:index_SW[-1]+1]=0               
    Hseg_LW = np.real(np.fft.ifft(S_eta_LW))     #inverted FFT eta time series    
    Hseg_LW = Hseg_LW[:seg_len]

#==============================================================================    
    # add segments of prescribed window length (M-Noverlap = 15 min)
    
    #unfiltered eta, converted to total elevation (including tide signal)
    if q==0:
        H[q:M-int(Noverlap/2)] = Hseg[:M-int(Noverlap/2)]+trend[:M-int(Noverlap/2)]  #include data at start of first segment
    elif q+M>=N:
        H[q+int(Noverlap/2):q+seg_len] = Hseg[int(Noverlap/2):]+trend[int(Noverlap/2):]   #limit segment to valid range at end of time series
    else:
        H[q+int(Noverlap/2):o-int(Noverlap/2)] = Hseg[int(Noverlap/2):-int(Noverlap/2)]+trend[int(Noverlap/2):-int(Noverlap/2)]
    
    #short waves (not including tide signal)
    if q==0:
        H_SW[q:M-int(Noverlap/2)] = Hseg_SW[:M-int(Noverlap/2)]  #include data at start of first segment
    elif q+M>=N:
        H_SW[q+int(Noverlap/2):q+seg_len] = Hseg_SW[int(Noverlap/2):]   #limit segment to valid range at end of time series
    else:
        H_SW[q+int(Noverlap/2):o-int(Noverlap/2)] = Hseg_SW[int(Noverlap/2):-int(Noverlap/2)]
    
    #long waves (including tidal signal)
    if q==0:
        H_LW[q:M-int(Noverlap/2)] = Hseg_LW[:M-int(Noverlap/2)]+trend[:M-int(Noverlap/2)]   #include data at start of first segment
    elif q+M>=N:
        H_LW[q+int(Noverlap/2):q+seg_len] = Hseg_LW[int(Noverlap/2):]+trend[int(Noverlap/2):]   #limit segment to valid range at end of time series
    else:
        H_LW[q+int(Noverlap/2):o-int(Noverlap/2)] = Hseg_LW[int(Noverlap/2):-int(Noverlap/2)]+trend[int(Noverlap/2):-int(Noverlap/2)]

#reshape to match length pt time series
H = H[:m] +zsens #convert to m TAW
H_SW = H_SW[:m]
H_LW = H_LW[:m] +zsens #convert to m TAW

#insert NaN values from original pressure time series
H_withnan[notnan.flatten()] = H
H = H_withnan
H_SW_withnan[notnan.flatten()] = H_SW
H_SW = H_SW_withnan
H_LW_withnan[notnan.flatten()] = H_LW
H_LW = H_LW_withnan

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# output elevation file
if write_output:
    etafile =os.path.join(basedir,'eta',location+'_eta_'+'%i' %fs+'Hz_'+pd.to_datetime(start_time).strftime('%Y_%m_%d')+'_'+pd.to_datetime(end_time).strftime('%Y_%m_%d')+output_suffix+ '.dat')
    np.savetxt(etafile, H_withnan, delimiter=",")
    etafile_SW =os.path.join(basedir,'eta',location+'_eta_SW_'+'%i' %fs+'Hz_'+pd.to_datetime(start_time).strftime('%Y_%m_%d')+'_'+pd.to_datetime(end_time).strftime('%Y_%m_%d')+output_suffix+ '.dat')
    np.savetxt(etafile_SW, H_SW_withnan, delimiter=",")
    etafile_LW =os.path.join(basedir,'eta',location+'_eta_LW_'+'%i' %fs+'Hz_'+pd.to_datetime(start_time).strftime('%Y_%m_%d')+'_'+pd.to_datetime(end_time).strftime('%Y_%m_%d')+output_suffix+ '.dat')
    np.savetxt(etafile_LW, H_LW_withnan, delimiter=",")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plotting
    
H_times = pd.date_range(pd.to_datetime(start_time),pd.to_datetime(start_time)+(len(H)-1)*pd.Timedelta(1/fs,'s'),freq='%i' %int(1000/fs) +'L')
eta=pd.DataFrame({'eta':H},index=H_times)
plt.plot(eta)
eta_SW=pd.DataFrame({'eta':H_SW},index=H_times)
plt.plot(eta_SW)
eta_LW=pd.DataFrame({'eta':H_LW},index=H_times)
plt.plot(eta_LW)

pres = pd.DataFrame({'p': pt_np}, index=time)
pt_5min = pres.resample('5T',loffset='2.5T').mean()
n_edge = 10*fs
pt_5min = pd.concat([pd.DataFrame({'p':pres[:n_edge].mean()[0]},index=[pres.index[0]]),pt_5min])
pt_5min = pd.concat([pt_5min,pd.DataFrame({'p':pres[-n_edge:].mean()[0]},index=[pres.index[-1]])])
pt_5min = pt_5min.resample(str(int(1000/fs))+'L').asfreq()
pt_swl = pt_5min.interpolate(method='polynomial',order=3) #polynomial interpolation   
pt_zero = pres-pt_swl

plot_p([[pt_zero,'pressure'],[eta_SW,'eta']],[pd.to_datetime('2022-01-30 00:00'),pd.to_datetime('2022-01-30 03:00')],None)
plt.plot(pt_zero/(rho*g))
plt.plot(eta_SW,'r:')
