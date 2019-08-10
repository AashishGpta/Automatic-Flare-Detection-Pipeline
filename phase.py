import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import pandas as pd
from astropy.stats import LombScargle
from scipy.optimize import curve_fit
def cos(x,a0,a1,f0,p0):
#     global freq
    return(a0 + a1*np.cos(2*np.pi*f0*(x+p0)))

def phase(kic):
    files = []
    direc = str(kic)
    for i in os.listdir(direc):
        if i[-12:] == 'slc_gulu.txt' and i[0]!= 'F':
            files.append(i)
    
    df_big = pd.DataFrame()
    for fname in files:
        df_big = pd.concat([df_big,pd.read_csv(direc + '//' + fname,sep = ' ',skiprows = 1,header = None)])
    df_big.columns = ['time','flux','err','original1','original2','straight','fit']
    df_big = df_big.sort_values(by = 'time')
    df_big.index = range(len(df_big))
    
    
    ls = LombScargle(df_big.time, df_big.straight, df_big.err)
    frequency, power = ls.autopower(minimum_frequency = (df_big.time.iloc[-1] - df_big.time[0])**-1,maximum_frequency = 100)#,nyquist_factor=(np.max(t)-p.min(t))/3)
    popt, pcov = curve_fit(cos, df_big.time, df_big.straight, [np.mean(df_big.straight),np.std(df_big.straight),frequency[np.argmax(power)],0],sigma = df_big.err,bounds = ([-np.inf,0,0,0],[np.inf,np.inf,np.inf,2*np.pi]))#,maxfev = 10000)
    y = cos(np.array(df_big.time),*popt)
    period = popt[2]**-1
    np.save(direc + '//' + direc + '_data_details.npy',np.array([period,((3*np.pi*np.std(df_big.straight-y))/(2*len(y)*abs(popt[1])))*(popt[2]**2),popt[1],np.sqrt(np.diag(pcov)[1])]))
    
    
    tt = np.arange(df_big.time[0],df_big.time.iloc[-1],20)
    tt = np.append(tt,df_big.time.iloc[-1]+1)
    
    cosy = []
    phase = []
    for i in range(len(tt)-1):
        df_temp = df_big[(df_big.time>=tt[i]) & (df_big.time<tt[i+1])]
        if len(df_temp)>0:
    #         check = pd.concat([check,df_temp])
            popt, pcov = curve_fit(cos, df_temp.time, df_temp.straight, [0,5*np.std(df_temp.straight),period**-1,0],sigma = df_temp.err,bounds = ([-np.inf,0,0,0],[np.inf,np.inf,np.inf,2*np.pi]))#,maxfev = 10000)
            cosy.extend(cos(np.array(df_temp.time),*popt))
            period = popt[2]**-1
            phase.extend(((df_temp.time+popt[3])%period)/period)
    
    df_big = df_big.assign(phase = phase,cos = cosy)
    
    
#    np.savetxt(direc + '//' + direc + '_df_big.txt',df_big,fmt = '%f')
    return(df_big)
    
