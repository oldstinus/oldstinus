#******************************************************************************
#  This script reads raw pressure time series, replaces data gaps and unphysical pressure recordings 
#  and fits z-position and density based on tidal elevations.
#  The converted pressure files can be exported for further processing in p_to_eta.py
#  Author:  Dieter Vanneste
#  March 2023
#******************************************************************************
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math
from scipy import signal
from scipy.signal import find_peaks, butter, peak_prominences, lfilter
from cycler import cycler


def clean_pressure(p_pd,t_window,dp_tres,fs,ts,te,fileW_gap):

    window = t_window #select window length for averaging pressure sample [min]
    p_pd_clean = p_pd.copy()
    #find false data lines (>=1min) corresponding with unphysical pressure drops
    
    p_pd_avg = p_pd.resample('%i' %t_window+'T').mean() 
    p_pd_avg_clean = p_pd_avg.copy()
    p_np = np.array(p_pd_avg_clean['p(Pa)'])   
    p_np_diff = np.diff(p_np)
    
    #peaks, _ = find_peaks(p5diff) #select all peaks
    #prominences = peak_prominences(p5diff,peaks)[0]
    #peaks, _ = find_peaks(p5diff,height=2000,prominence=np.quantile(prominences,0.95))  
    peaks_neg, _ = find_peaks(-p_np_diff,height=dp_tres) #find negative gradients = pressure dropping away from tide signal
#    prominences_neg = peak_prominences(-p_np_diff,peaks_neg,wlen=20)
    peaks, _ = find_peaks(p_np_diff,height=dp_tres)  #find positive gradients = pressure going back to tide signal
#    prominences = peak_prominences(p_np_diff,peaks,wlen=20)
    
    #replace with NaN values
    i_peaks,i_peaks_neg = 0,0
    cycle = 0      
    while (i_peaks_neg<len(peaks_neg)) and (i_peaks<len(peaks)):
        flag_clean = 1
    #    print('cycle '+str(cycle)+' ipeaks_neg='+str(i_peaks_neg)+', ipeaks='+str(i_peaks))
        xnan_s = peaks_neg[i_peaks_neg]        
        c=1
        while abs(p_np_diff[xnan_s-c])>dp_tres:
            c+=1
        xnan_s = xnan_s-c    
        xnan_e = peaks[i_peaks]
        c=1
        while abs(p_np_diff[xnan_e+c])>dp_tres:
            c+=1
        xnan_e = xnan_e+c           
        if xnan_e>xnan_s: #pair marking start and end of pressure drop        
            i_peaks+=1
            i_peaks_neg+=1        
        else: #single pos peak
            c=1
            if not np.isnan(p_np_diff[xnan_e-c]): #the negative pressure drop is below the gradient treshold,no need to remove
                flag_clean =0
            else:    
                while not np.isnan(p_np_diff[xnan_e-c]):
                    c=c+1
            xnan_s = xnan_e-c
            i_peaks+=1    
        if flag_clean==1:
            for j in range(xnan_s+1,xnan_e+1):               
                p_pd_avg_clean.iloc[j]=np.nan  #replace false data by NaN                
                tnan_s,tnan_e = p_pd_avg.index[xnan_s],p_pd_avg.index[xnan_e]
            p_pd_clean[tnan_s:tnan_e] = np.nan
            print('false data: '+str(tnan_s)+' - '+str(tnan_e))
            fileW_gap.write('false data: '+str(tnan_s)+' - '+str(tnan_e) + '\n' )
        cycle+=1
    
    #fill missing data with NaN values
    freq_out = 1000/fs
    p_pd_clean = p_pd_clean.reindex(pd.date_range(ts,te,freq='%i' %freq_out+'L'),fill_value=np.nan)  

    p_pd_clean.index.names = ['time']
    
    #remove spikes (unphysical data with short duration (a few succesive recordings))
    p_single = np.array(p_pd_clean['p(Pa)'])
    p_single_diff=np.diff(p_single)

    singlepeaks, _ = find_peaks(p_single_diff,height=5000) 
    for x in singlepeaks:
        p_pd_clean.iloc[x] = np.nan  #remove peaks
    
    valid = 100*len(p_pd_clean.dropna())/len(p_pd_clean)
    fileW_gap.write('%.1f' %valid +'% valid data'+'\n')
    
    return p_pd_clean

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_p(p,window,ylim,output_dir,location):
    
    custom_cycler = (cycler(color=['k','r', 'b', 'y', 'g'])*cycler(linestyle=['-']))
#    if window == 'all':
#        ts = pd.to_datetime('2022-01-01 00:00')
#        te = pd.to_datetime('2022-02-01 00:00')
#    else:
    ts,te = window[0],window[1]
    
    fig, ax = plt.subplots(figsize=cm2inch(20,15))
    plt.rc('axes', prop_cycle=custom_cycler)
    labelticksize='8'
    axtitlesize='10'
    legendfontsize='7'
    
    for pres in p:
        ax.plot(pres[0][ts:te],label=pres[1],)

    ax.set_xlabel('time [UTC]',fontsize=axtitlesize,labelpad=10)
    ax.set_ylabel('pressure [Pa]',fontsize=axtitlesize,labelpad=10)
    
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
      
    ax.tick_params( direction='out', length=4, width=1, labelsize=labelticksize)    
    ax.grid(True, which='major', color='grey', linestyle=':')
    ax.set_title('druksensor '+location,fontsize=axtitlesize)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,0.1), bbox_transform=ax.transAxes,fontsize=legendfontsize)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,location+'_p_'+ts.strftime('%Y-%m-%d %H:%M').replace(':','-')
                             +'_'+te.strftime('%Y-%m-%d %H:%M').replace(':','-')+'.png'),dpi=300)
    plt.close(fig)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main(location,year,list_months,fs_out,output_p_series,select_drive,calibrate_window):

    ID_locations = (['Jachthaven',2001375.64,-248.28,'OSB'],
             ['Belwind',2006781.17,-572.54,'OSD'],
             ['Demeysluis',2011536.81,-925.31,'OSE'],
             [ 'Station',2005284.13,-417.10,'OSA'],
             ['Vistrap',1997911.26,-738.68,'OSC'])
    # location = 'Belwind'
    # year='2020'
    # list_months=['12']
    # month='12'
    # fs_out=2
    # select_drive='C'
    
    datadir = r'P:\18_053-AnalgolfmOstn\2_Input_gegevens'
    fs = 2  #original sample freq raw V data [Hz]
    length_data_line = 60*fs #row-based number of data points in raw data file
    read_input = True  #flag to avoid re-reading data 
    global fileW_gap,output_dir
    
    if select_drive=='C':
        output_dir = os.path.join(r'C:\Users\vannesdi\Documents\work files\18_053\P_series',location)
    elif select_drive=='P':
        output_dir = os.path.join(r'P:\18_053-AnalgolfmOstn\3_Uitvoering\03. Wave analysis\P_series',location)    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # input 5min-tide gauge data    
    datafile_tide =os.path.join(datadir,'tide_data',year+'.txt')  
    data_tide = pd.read_csv(os.path.join(datadir,'tide_data',datafile_tide),index_col=None,header=0,sep='\t',
                            names = ['time','mTAW'])
    data_tide['time']= pd.to_datetime(data_tide['time'])
    data_tide['time'] = data_tide['time'].dt.tz_localize(None)
    data_tide['mTAW'] = 0.01*data_tide['mTAW']    
       
    for i in range(0,len(ID_locations)):
        if ID_locations[i][0] == location:
            a_cal = ID_locations[i][1] 
            b_cal = ID_locations[i][2]
            sensorID = ID_locations[i][3]
    
    for month in list_months:                 
        
        fileW_gap = open(os.path.join(output_dir,location+'_data_gap_'+month+'_'+year+'.txt'), 'w')  #create output files
        fileW_gap.write('data gap from'+' - '+'till' +'\n') 
        print('processing '+location+', '+month+'-' +year)
    
        if read_input==False and int(year)>=2019 and int(month)>=6: 
            read_input=True
        if read_input:
            if year=='2018':
                if int(month)<10:    
                    read_V_file = [os.path.join(datadir,'raw data druksensoren',
                   '20181009_data_golfmetingen_20180110_20181008',year+month+'_'+location+'_'+ 'Golfwaarden.dat')]
                else:
                    read_V_file = [os.path.join(datadir,'raw data druksensoren','2018_10_01-2019_05_31',
                                                location+'_'+ 'Golfwaarden.dat')]
            if year=='2019' and int(month)<=6:
                if int(month)<6:    
                    read_V_file = [os.path.join(datadir,'raw data druksensoren','2018_10_01-2019_05_31',
                                                location+'_'+ 'Golfwaarden.dat')]
                elif int(month)==6: 
                    read_V_file = [os.path.join(datadir,'raw data druksensoren',
                                                year+'_'+month,year+month+'_'+location+'.dat')]
            else:
                input_list = os.listdir(os.path.join(datadir,'raw data druksensoren',year+'_'+month))
                read_V_file = [os.path.join(datadir,'raw data druksensoren',year+'_'+month,i) 
                               for i in input_list if sensorID in i]
            if not read_V_file:
                raise Exception('missing measurement data, please check')
                    
        rawpdata_all = []
        for rpf in read_V_file:
            data = []    
            try:
                fileR = open(rpf, 'r')
            except:
                print('WARNING - File not present: ' + os.path.join(rpf))
                print('Done')    
            else:
                data = fileR.read()
                fileR.close()
                data = data.split('\n')    
            #check for header
            header,count=0,0
            while 'Smp' not in data[count] and count < len(data)-1:
                count+=1
                if 'Smp' in data[count]:
                    header = count
                    break
            data = data[header+1:]  #remove header lines
            
            # Read the csv file
            if header==0:
                rawpdata = pd.read_csv(os.path.join(rpf),index_col=None,header=None,
                                       names=['time','index','valid']+['p'+ '%i' %i for i in range(1,121)])
            else:
                rawpdata = pd.read_csv(os.path.join(rpf),index_col=None,header=header,
                                       names=['time','index','valid']+['p'+ '%i' %i for i in range(1,121)])
            rawpdata = rawpdata.replace([-9999,'NAN'],[np.nan,np.nan])  #identify erroneous measurements
            rawpdata = rawpdata.dropna(subset=['p'+ '%i' %i for i in range(1,121)],how='all') #remove entire erroneous measurement lines
            rawpdata['time']=pd.to_datetime(rawpdata['time'])
        #        rawpdata['time'] = [i.tz_localize('UTC') for i in rawpdata['time']]
            rawpdata_all.append(rawpdata)
        rawpdata = pd.concat(rawpdata_all,ignore_index=True)
        rawpdata = rawpdata.drop_duplicates()
        
        if (year=='2018' and int(month)>=10) or (year=='2019' and int(month)<6):
            read_input=False  #avoid re-reading input data file 
            
    #==============================================================================
    
    start_time =year+'-'+month+'-01 00:00:00'  
    if month !='12':
        endmonth=str(int(month)+1)
        end_time =year+'-'+endmonth+'-01 00:00:00' 
    else:
        endyear=str(int(year)+1)
        end_time =endyear+'-01-01 00:00:00'
    if calibrate_window != 'all':          
        if calibrate_window[0] != 'start':
            start_time =year+'-'+month+'-'+str(calibrate_window[0]).zfill(2) +' 00:00:00'
        elif calibrate_window[1] != 'end': 
            end_time =year+'-'+month+'-'+str(calibrate_window[1]).zfill(2) +' 00:00:00'
    
    ts=pd.to_datetime(start_time)
    #    ts = ts.tz_localize('UTC')
    te=pd.to_datetime(end_time)    
    #    te = te.tz_localize('UTC')
         
    #read specified time window            
    i_start = (np.abs(rawpdata['time']-ts)).idxmin()
    if rawpdata['time'][i_start].month==int(month)-1:  # ensure first record in processed month
       i_start+=1
    i_end = (np.abs(rawpdata['time']-te)).idxmin()    
    if rawpdata['time'][i_end].month==int(month):  # include last time record on last day of month
       i_end+=1


    #==============================================================================
    
    #search data gaps (missing time lines of >=1 min duration)
    rawpdata_sub=pd.DataFrame()
    rawpdata_sub['time'] = rawpdata['time'][i_start:i_end+1].copy()
    deltas = rawpdata_sub['time'].diff()[1:]
    gaps = deltas[deltas > timedelta(seconds=length_data_line/fs)]    
    list_gaps_index = [i_start]+[i for i in gaps.index]+[i_end+1]
    
    if len(list_gaps_index)==2:
        if rawpdata_sub['time'][i_start] - ts > timedelta(minutes=1):
            fileW_gap.write(str(ts) + ' - ' + str(rawpdata_sub['time'][i_start]) + '\n' )
        if te-rawpdata_sub['time'][i_end-1] > timedelta(minutes=1):
            fileW_gap.write(str(rawpdata_sub['time'][i_end-1]) + ' - ' + str(te) + '\n' )    
        else:
            fileW_gap.write('no data gap in '+ month +'-'+year +'\n')
    elif len(list_gaps_index)>2:
        if rawpdata_sub['time'][i_start]-ts > timedelta(minutes=1):
            fileW_gap.write(str(ts) +' - '+str(rawpdata_sub['time'][i_start]) + '\n' )
        for i in range(1,len(list_gaps_index)-1):
            fileW_gap.write(str(rawpdata_sub['time'][list_gaps_index[i]-1]+timedelta(minutes=1)) +
                            ' - '+str(rawpdata_sub['time'][list_gaps_index[i]]) + '\n' )
        if te-rawpdata_sub['time'][list_gaps_index[-1]-1] > timedelta(minutes=1):        
            fileW_gap.write(str(rawpdata_sub['time'][i_end-1]) + ' - ' + str(te) + '\n' )
    
    # create numpy array with p data from start time to end time, considering data gaps (=non-recorded data line)
    time_pd = pd.DataFrame()
    len_pd=[]
    p_pd_fs_out = []
    for t in range(0,len(list_gaps_index)-1):
        j,k = list_gaps_index[t],list_gaps_index[t+1]        
        df_add = pd.DataFrame({'time':pd.date_range(rawpdata_sub['time'][j],rawpdata_sub['time'][k-1]+
                                                    timedelta(seconds=59.5), freq='%i' %int(1000/fs) +'L')})
        len_pd.append(len(df_add))
        time_pd = pd.concat([time_pd,df_add],ignore_index=True)
        
        if fs_out > fs:
        #output at fs_out
            i_s = (np.abs(rawpdata['time']-pd.to_datetime(df_add['time'].iloc[0]))).idxmin()
            i_e = (np.abs(rawpdata['time']-pd.to_datetime(df_add['time'].iloc[-1]))).idxmin()
            data_np = np.array(rawpdata[i_s:i_e][['p'+ '%i' %i for i in range(1,121)]],dtype='float')      
            data_np = data_np.flatten()
            data_np = a_cal*(0.01*data_np.astype(np.float64))+b_cal    #convert from measurement unit 10 mV to V to Pa   
            p_part = pd.DataFrame({'p(Pa)':data_np},
            index=pd.date_range(rawpdata_sub['time'][i_s],rawpdata_sub['time'][i_e-1]+timedelta(seconds=59.5),
                                freq='%i' %int(1000/fs) +'L'))    
            p_part = p_part.resample(str(int(1000/fs_out))+'L').asfreq().interpolate(method='polynomial',order=3)  
            pd.concat([p_pd_fs_out,p_part])
            # p_pd_fs_out.append(p_part)
      
    data_np = np.array(rawpdata[i_start:i_end][['p'+ '%i' %i for i in range(1,121)]],dtype='float')                
    data_np = data_np.flatten()
    data_np = a_cal*(0.01*data_np.astype(np.float64))+b_cal    #convert from measurement unit 10 mV to V to Pa                
    if len(data_np) != len(time_pd):
        raise Exception('error in data gap finding, please check')
    else:
        p_pd = pd.DataFrame({'p(Pa)': data_np}, index=time_pd['time'])
        
    #Make a csv file of time and pressure as an input file for p_to_eta.py
    p_pd_clean = clean_pressure(p_pd,1,2000,fs,ts,te,fileW_gap)
    if fs_out > fs:
        p_pd_fs_out = pd.concat(p_pd_fs_out)
        p_pd_fs_out_clean = clean_pressure(p_pd_fs_out,1,2000,fs_out,ts,te)
    
    if output_p_series:
        p_pd_clean.to_csv(os.path.join(output_dir,location+'_'+ts.strftime('%Y_%m_%d')+'_'+te.strftime('%Y_%m_%d')
                                       + '_'+'%i' %fs+'Hz_clean.dat'),
                          index = True, header=True,sep='\t',float_format='%.2f')
        if fs_out > fs:
            p_pd_fs_out_clean.to_csv(os.path.join(output_dir,location+'_'+ts.strftime('%Y_%m_%d')
                                                  +'_'+te.strftime('%Y_%m_%d')+ '_'+'%i' %fs_out+'Hz_clean.dat'),
                                     index = True, header=True,sep='\t',float_format='%.2f')
    
    
    #==============================================================================
    #perform regression analysis on tide and pressure data
            
    fileW_fit = open(os.path.join(output_dir,location+'_data_fit_'+month+'_'+year+'.txt'), 'w') 
    fileW_fit.write('period' + '\t' + 'rho [kg/m^3]' + '\t'+'Z [mTAW]' + '\n')
    
    # select tide gauge data window
    i_start1,i_end1 = (np.abs(data_tide['time']-ts)).idxmin(),(np.abs(data_tide['time']-te)).idxmin()        
    tide = pd.DataFrame({'tide(mTAW)':data_tide[i_start1:i_end1]['mTAW'].values},
                        index=data_tide['time'][i_start1:i_end1].values)
    
    # combine pressure and tide data
    p_pd_5min=p_pd_clean.dropna().resample('5T').mean() 
    merge=pd.merge(p_pd_5min,tide, how='inner', left_index=True, right_index=True)   
    
    fig,ax = plt.subplots(figsize=cm2inch(20,17));
    plt.title(location+',  '+ year+'-'+ month, fontsize=20)
    plt.xlabel('tide [m TAW]',fontsize=18)
    plt.ylabel('static pres. [kPa]',fontsize=18)
    #    plt.scatter(t_np, p5/1000,s=2,c='k')
    plt.scatter(merge['tide(mTAW)'], merge['p(Pa)']/1000,s=2,c='k')
    x,y = merge['tide(mTAW)'].values, merge['p(Pa)'].values
    
    idx = np.isfinite(x) & np.isfinite(y)
    z = np.polyfit(x[idx],y[idx], 1)
    rho= z[0]/9.81
    zpos=-z[1]/(9.81*rho)
    p = np.poly1d(z)
    fileW_fit.write(month +'-'+year + '\t' + '%.1f' %rho + '\t' + '%.2f' %zpos +'\n')
    
    plt.plot(x,p(x)/1000,"r--")
    plt.text(2,25,"ρ=%.2f kg/m³"%(rho), ha='left',fontsize=18)
    plt.text(2,20,"Z=%.2f m TAW"%(zpos), ha='left',fontsize=18)
    
    #    xmin,xmax = round(0.5*math.floor(min(t_np)/0.5),1),round(0.5*math.ceil(max(t_np)/0.5),1)
    xmin,xmax=-1,6 #fixed range (m TAW)
    plt.xticks(np.arange(xmin,xmax+1,1),fontsize=14)
    plt.xlim(xmin,xmax)
    ymin,ymax = 0,round(10*math.ceil(max(merge['p(Pa)']/1000)/10),1)
    plt.yticks(fontsize=14)
    plt.ylim(ymin,ymax)
    
    plt.tight_layout()    
    plt.savefig(os.path.join(output_dir,location+'_'+ year+'-'+ month+ '.png'),dpi=300)   # save the figure to file
    plt.close(fig)
    
    fileW_gap.close()
    fileW_fit.close()
    
    return p_pd,p_pd_clean,output_dir

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']    

# list_stations = ['Jachthaven']
# #list_stations = ['Station']
# #list_stations = ['Vistrap']
# list_stations = ['Belwind']

# if __name__ == '__main__':
# #    def main(location,year,list_months,fs_out,output_p_series,select_drive):    
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2019',['01'],2,True,'C')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],[pd.to_datetime('2019-01-08 00:00'),pd.to_datetime('2019-01-09 00:00')],None,output_dir,loc)
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],[pd.to_datetime('2019-01-27 18:00'),pd.to_datetime('2019-01-28 18:00')],None,output_dir,loc)

# if __name__ == '__main__':
# #    def main(location,year,list_months,fs_out,output_p_series,select_drive):    
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2019',['12'],2,True,'C')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],[pd.to_datetime('2019-12-09 03:00'),pd.to_datetime('2019-12-10 03:00')],None,output_dir,loc)

# if __name__ == '__main__':
# #    def main(location,year,list_months,fs_out,output_p_series,select_drive):    
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2021',['12'],2,True,'C')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
#                [pd.to_datetime('2021-12-01 18:00'),pd.to_datetime('2021-12-02 18:00')],None,output_dir,loc)
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
#                [pd.to_datetime('2021-12-10 06:00'),pd.to_datetime('2021-12-11 06:00')],None,output_dir,loc)

# if __name__ == '__main__':
#     list_stations = ['Jachthaven','Vistrap','Station','Demeysluis']   
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2022',['01'],2,True,'C','all')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
#            [pd.to_datetime('2022-01-31 00:00'),pd.to_datetime('2022-02-01 00:00')],None,output_dir,loc)
#     p_pd,p_pd_clean,output_dir = main('Belwind','2022',['01'],2,True,'C',[6,'end'])

# if __name__ == '__main__':
#     list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2022',['02'],2,True,'C','all')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
#                [pd.to_datetime('2022-02-06 18:00'),pd.to_datetime('2022-02-07 12:00')],None,output_dir,loc)
    
# if __name__ == '__main__':
#     list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2022',['09'],2,True,'C','all')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
#                [pd.to_datetime('2022-09-16 18:00'),pd.to_datetime('2022-09-17 18:00')],None,output_dir,loc)    
    
# if __name__ == '__main__':
#     list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
#     for loc in list_stations:    
#         p_pd,p_pd_clean,output_dir = main(loc,'2020',['03'],2,True,'C','all')
#         plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
#                 [pd.to_datetime('2020-03-28 21:00'),pd.to_datetime('2020-03-29 21:00')],None,output_dir,loc)   

if __name__ == '__main__':
    list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
    for loc in list_stations:    
        p_pd,p_pd_clean,output_dir = main(loc,'2020',['08'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2020-08-30 03:00'),pd.to_datetime('2020-08-30 21:00')],None,output_dir,loc)            
        
if __name__ == '__main__':
    list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
    for loc in list_stations:    
        p_pd,p_pd_clean,output_dir = main(loc,'2020',['09'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2020-09-27 03:00'),pd.to_datetime('2020-09-27 18:00')],None,output_dir,loc) 
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2020-09-25 12:00'),pd.to_datetime('2020-09-26 03:00')],None,output_dir,loc) 
        
if __name__ == '__main__':
    list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
    list_stations = ['Station']  
    for loc in list_stations:    
        p_pd,p_pd_clean,output_dir = main(loc,'2020',['12'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2020-12-01 06:00'),pd.to_datetime('2020-12-01 18:00')],None,output_dir,loc) 
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2020-12-24 09:00'),pd.to_datetime('2020-12-25 09:00')],None,output_dir,loc)
        
if __name__ == '__main__':
    list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
    for loc in list_stations:    
        p_pd,p_pd_clean,output_dir = main(loc,'2021',['03'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2021-03-16 18:00'),pd.to_datetime('2021-03-17 09:00')],None,output_dir,loc) 
        
if __name__ == '__main__':
    list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
    for loc in list_stations:    
        p_pd,p_pd_clean,output_dir = main(loc,'2021',['04'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2021-04-05 12:00'),pd.to_datetime('2021-04-06 00:00')],None,output_dir,loc) 
    p_pd,p_pd_clean,output_dir = main('Jachthaven','2021',['04'],2,True,'C',[15,'end'])
    plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
               [pd.to_datetime('2021-04-05 12:00'),pd.to_datetime('2021-04-06 00:00')],None,output_dir,loc) 
    
if __name__ == '__main__':
    list_stations = ['Jachthaven','Belwind','Vistrap','Station','Demeysluis']   
    for loc in list_stations:    
        p_pd,p_pd_clean,output_dir = main(loc,'2022',['03'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2022-03-31 18:00'),pd.to_datetime('2022-04-01 00:00')],None,output_dir,loc) 
        p_pd,p_pd_clean,output_dir = main(loc,'2022',['04'],2,True,'C','all')
        plot_p([[p_pd,'raw pressure'],[p_pd_clean,'cleaned']],
                [pd.to_datetime('2022-04-01 00:00'),pd.to_datetime('2022-04-01 12:00')],None,output_dir,loc) 