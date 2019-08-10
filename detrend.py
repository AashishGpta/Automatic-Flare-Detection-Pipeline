import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from astropy.table import Table

from astropy.stats import LombScargle
from scipy.optimize import curve_fit

from scipy.interpolate import interp1d

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='both',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)                     & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


import statsmodels.api as sm


def fourier(x, *a):
    ret = a[0]
    for deg in range(int((len(a)-1)/2)):
#         print(deg,a[deg*2 + 1],freq[deg],)
        ret += a[deg*2 + 1] * np.cos( 2*np.pi*freq[deg]*x + a[deg*2 + 2])
    return ret

def straight1(x,*a):
    ret = 0
    for deg in range(len(a)):
        ret += a[deg]*(x**deg)
    return(ret)# + a4*(x**4) + a5*(x**5))


from sklearn.metrics import r2_score

import kplr

client = kplr.API()

def detrend(kic):
    
    star = client.star(kic)
    
    lcs = star.get_light_curves()
    
    
    sclc = []
    for lc in lcs:
        if lc.filename[-8:-5] == 'slc':
            sclc.append(lc)
            
    if len(sclc) == 0:
        return(False)
    else:
        print("Short cadence files: ",len(sclc))
    
        global freq
        plots = False
        plotsf = False
        # df_exp_big = pd.DataFrame()
        direc = str(kic)
        try:
            os.mkdir(direc)    # Add condition to not run if directory exists
        except Exception:
            pass
        for fname in sclc:
        # for fname in ['kplr009726699-2012121044856_slc.fits']:
        # for i in ['kplr009726699-2012032013838_slc.fits']:
            freq = []
            print("FILENAME: ",fname.filename[-32:])
            print("Reading Data")
            afile = fname.open()[1]
            single_file = [fname.filename[-32:],afile.header['DATE-OBS'],afile.header['DATE-END'],afile.header['TSTART'],afile.header['TSTOP']]
        #     data = fits.getdata(direc + '\\' + i, 1)
            t = Table(afile.data)
        #     plt.plot(np.array(t['TIME']),np.array(t['SAP_FLUX']))
        #     plt.show()
        #     time = np.aoppend(time,np.array(t['TIME']))
        #     #print(time[-1]-time[-2],i)
        #     flux = np.append(flux,np.array(t['PDCSAP_FLUX']))
            #if (t['TIME'][0] < 1130) and (t['TIME'][-1] > 1130):
            df_exp1 = pd.DataFrame([t['TIME'],t['PDCSAP_FLUX'],t['PDCSAP_FLUX_ERR'],t['SAP_QUALITY']]).T
            df_exp1.columns =['time','flux','err','flag']
            df_exp1 = df_exp1[(df_exp1.time >= single_file[-2]) & (df_exp1.time <= single_file[-1]) & (df_exp1.flux <= 1e+14)]
            #df_filtered = df.query(1>30000)
            #df
            df_exp1.dropna(inplace = True)
            df_exp1 = df_exp1.sort_values(by = 'time')
            df_exp1.index = range(len(df_exp1))
            
            a = np.array(df_exp1.flux) 
            thresh= 1.5*np.std(a) #+ np.median(a)
        #     print(thresh)
            mask = a[1:-1]-np.minimum(a[2:],a[:-2]) > thresh
            means = np.mean([a[2:],a[:-2]],axis = 0)
            df_exp1.flux[1:-1][mask] = means[mask]
        
        #     single_file.append(df_exp1.time[int(len(df_exp1)/2)])
        #     plt.plot(df_exp1.time,df_exp1.flux)
        #     plt.show()
            print("Detrending Data")
            #df_exp = df_exp1.assign()
        #     df_exp = df_exp1[((df_exp1['flag'] != 16) & (df_exp1['flag'] != 128) & (df_exp1['flag'] != 2048))][['time','flux','err']]#[(df_exp1.time > 1130) & (df_exp1.time < 1131.2)]
            df_exp = df_exp1[['time','flux','err']]
        
        #     plt.close()
            dt = np.array(df_exp.time)[1:]-np.array(df_exp.time)[:-1]
            if np.min(dt) > 0.01:    #  Can remove these cadence related conditions now
                gap = np.append([-1],np.where(dt >= 1))
                point_thresh = 1500
            else:
                point_thresh = 15000
                gap = np.append([-1],np.where(dt >= 0.2))
            gap = gap + 1
            gap = np.append(gap,len(df_exp))
        #     print(gap)
            y = np.empty(0)
            y1 = np.empty(0)
            for i in range(len(gap)-1):
                df_temp = df_exp.loc[gap[i]:gap[i+1]]
                try:
                    popt, pcov = curve_fit(straight1,df_temp.time, df_temp.flux, [np.mean(df_exp.flux),0],sigma = df_temp.err)
                    y1 = np.append(y1,straight1(np.array(df_exp.time[gap[i]:gap[i+1]]),*popt))            
                    popt, pcov = curve_fit(straight1,df_temp.time, df_temp.flux, list(popt) + [0]*min(int(len(df_temp)/point_thresh),2),sigma = df_temp.err)
                    y = np.append(y,straight1(np.array(df_exp.time[gap[i]:gap[i+1]]),*popt))
                except Exception as e:
                    print(e)
                    y = np.append(y,np.array(df_exp.flux[gap[i]:gap[i+1]]))
                    print(len(y))
            if plots:
                plt.plot(df_exp.time,df_exp.flux,label = 'Original')
                plt.plot(df_exp.time,y,label = 'Polynomial Fit')
                plt.legend()
            # #     plt.xlabel("Time (d)")
            # #     plt.ylabel("Flux")
            # #     plt.savefig("2.png")
                plt.show()
        
            df_exp = df_exp.assign(flux=df_exp['flux'].subtract(y))
        
        #     plt.plot(df_exp.time,df_exp.flux)
        # #     plt.xlabel("Time (d)")
        # #     plt.ylabel("Flux")
        # #     plt.savefig("3.png")
        #     plt.show()
        #     np.savetxt(direc + '//' + fname[:-5] + '_exp.txt',df_exp,fmt = '%f',header='Detrended')
        #     df_exp_big = pd.concat([df_exp_big,df_exp])
            df_gulu = df_exp.copy()
        #     a = np.array(df_gulu.flux) 
        #     thresh= 1.5*np.std(a) #+ np.median(a)
        # #     print(thresh)
        #     mask = a[1:-1]-np.minimum(a[2:],a[:-2]) > thresh
        #     means = np.mean([a[2:],a[:-2]],axis = 0)
        #     df_gulu.flux[1:-1][mask] = means[mask]
            temp_thresh = float(np.median(df_gulu.flux)+2*np.std(df_gulu.flux))
            df_temp = df_gulu[(df_gulu.flux < temp_thresh) & (df_gulu.flux > -temp_thresh)]
            fmax = 50
            fmin = max(1000,df_temp.time.iloc[-1]-df_temp.time.iloc[0])**-1
            
            ls = LombScargle(df_temp.time, df_temp.flux, df_temp.err)
            frequency, power = ls.autopower(minimum_frequency = fmin,maximum_frequency = fmax)#,nyquist_factor=(np.max(t)-p.min(t))/3)
            period = frequency[np.argmax(power)]**-1
        #     try:
        #         popt, pcov = curve_fit(cos, df_temp.time, df_temp.flux, [np.mean(df_temp.flux),amplitude_guess,frequency[np.argmax(power)],0],sigma = df_temp.err,bounds = ([-np.inf,0,0,0],[np.inf,np.inf,np.inf,2*np.pi]))#,maxfev = 10000)
        #         y = cos(np.array(df_gulu.time),*popt)
        #         #y = cos(np.array(df_temp.time),*popt)
        #     except Exception:
        #         y = [np.nan]*len(df_gulu)
        #         popt = [np.nan,np.nan,np.nan,np.nan]
                
        # #     freq_main = popt[2]
        #     period = popt[2]**-1
        #     single_file.append(period)
        
            df_gulu['original1'] = df_exp1.flux
            df_gulu['original2'] = df_exp.flux
        #     df_gulu['cos'] = y#df_exp1.flux
            df_gulu['straight'] = df_exp1.flux - y1#df_exp1.flux
        #     df_gulu['phase'] = ((df_gulu.time+popt[3])%period)/period
            df_gulu['fit'] = df_gulu['original1']-df_gulu['original2']
        
        #     single_file.append(gap)
            mask = power>4*np.std(power)
            power = power[mask]
            frequency = frequency[mask]
            powers_imp = sorted(power[detect_peaks(power,show = False)],reverse = True)
            for i in range(min(len(powers_imp),4)):
                freq.append(float(frequency[np.where(power == powers_imp[i])]))
            gap[1:-1] = gap[1:-1] + 1
            times = np.empty(0)
            for g in range(len(gap)-1):
                times = np.append(times,np.arange(df_exp.time[gap[g]],df_exp.time[gap[g+1]-1],min(max(7*period,1),10)))
                #times = [df_exp.time[0],df_exp.time[len(df_exp1)-1]]
        #             times[-1] != df_exp.time[gap[g+1]-1]:
                if len(times) > 1:
                    times = np.append(times[:-1],df_exp.time[gap[g+1]-1])
                else:
                    times = np.append(times,df_exp.time[gap[g+1]-1])
            times[-1] = times[-1] + 0.01
         
            amplitude_guess = np.std(df_gulu.flux)
        
        # if True:
            ignore = True
            n = 5
            j = 1
            while j<n:
                print(j)
        
            #     ls = periodic.LombScargleFast().fit(df_temp.time, df_temp.flux, df_temp.err)
            #     ls.optimizer.period_range = (0.1, (df_gulu.time[len(df_gulu)-1]-df_gulu.time[0])/2)
            #     freq = ls.find_best_periods(1)**-1
                if not ignore:
                    freq = []
                    ls = LombScargle(df_temp.time, df_temp.flux, df_temp.err)
                    frequency, power = ls.autopower(minimum_frequency = fmin,maximum_frequency = fmax)#,nyquist_factor=(np.max(t)-p.min(t))/3)
        #             freq_main = frequency[np.argmax(power)]
                    mask = power>4*np.std(power)
                    if not mask.any():
                        print(j,"Didn't work 1.")
                        break
                    power = power[mask]
                    frequency = frequency[mask]
                    powers_imp = sorted(power[detect_peaks(power,show = False)],reverse = True)
        #             print(freq)
                    for i in range(min(len(powers_imp),5)):
                        freq.append(float(frequency[np.where(power == powers_imp[i])]))
        #             print(freq)
                y = np.empty(0)
                for i in range(len(times)-1):
                    t1 = times[i]
                    t2 = times[i+1]
                    df_tempt = df_temp[(df_temp.time >= t1) & (df_temp.time < t2)]
                    try:
                        guess = [np.mean(df_temp.flux)] + [amplitude_guess,0]*len(freq)
                        popt, pcov = curve_fit(fourier, df_tempt.time, df_tempt.flux, guess,sigma = df_tempt.err)#,maxfev = 10000)
                        y = np.append(y,fourier(np.array(df_gulu[(df_gulu.time >= t1) & (df_gulu.time < t2)].time),*popt))
                    except Exception:
    #                    print(i,e)
                        if len(df_tempt.time)<3:
        #                     print(1)
                            y = np.append(y,df_gulu[(df_gulu.time >= t1) & (df_gulu.time < t2)].flux)
                        else:
            #                 end = True
                            print(j,"Didn't work.2")
                            y = np.zeros(len(df_gulu))
                            break
                if plotsf:
                    plt.plot(df_gulu.time,df_gulu.flux,label = 'Original')
                    plt.plot(df_gulu.time,y,label = 'No.' +str(j) + ' Fourier Fit')
            #         plt.legend()
            #         plt.xlabel("Time (d)")
            #         plt.ylabel("Flux")
            #         plt.savefig(str(j)+".png")
                    plt.show()
                temp_thresh = float(np.median(df_gulu.flux)+2*np.std(df_gulu.flux))
                r2 = r2_score(df_gulu[(df_gulu.flux < temp_thresh) & (df_gulu.flux > -temp_thresh)].flux,y[df_gulu[(df_gulu.flux < temp_thresh) & (df_gulu.flux > -temp_thresh)].index])
        #         print(r2,0.15)
        #         print(chisquare(df_gulu[(df_gulu.flux < temp_thresh) & (df_gulu.flux > -temp_thresh)].flux,y[df_gulu[(df_gulu.flux < temp_thresh) & (df_gulu.flux > -temp_thresh)].index]))
                if r2 < 0.1:
                    print(j,r2,"Didn't work.3")
                    ignore = True
                    break
                else:
                    ignore = False
                times = np.concatenate(([times[0]],times[1:-1]+2*(j%2)-1,[times[-1]]))
                j = j + 1
                df_gulu = df_gulu.assign(flux=df_gulu['flux'].subtract(y),fit=df_gulu['fit'].add(y))
                
                #temp_thresh = float(np.median(df_gulu.flux)+2*np.std(df_gulu.flux))
                df_temp = df_gulu[(df_gulu.flux < temp_thresh) & (df_gulu.flux > -temp_thresh)]
        #         times = np.append((times+2)[:-1],times[-1])  Can do something for dynamic change in times
        #     print(freq_main)
            if j >1:
                temp_thresh = 0.5*float(np.std(df_gulu.flux)) # Can be 1*flo.. too
            else:
                temp_thresh = 3*float(np.std(df_gulu.flux))  
            med = np.median(df_gulu.flux)
            df_temp = df_gulu[(df_gulu.flux < med + temp_thresh) & (df_gulu.flux > med -temp_thresh)]
            f = interp1d(df_temp.time,sm.tsa.filters.hpfilter(df_temp.flux, lamb=1e6)[1],kind='linear',fill_value = 'extrapolate')
            if plots:
                plt.plot(df_gulu.time,df_gulu.flux,label = 'Original')
                plt.plot(df_gulu.time,f(df_gulu.time),label = 'HP Filter')
                plt.legend()
            #     plt.xlabel("Time (d)")
            #     plt.ylabel("Flux")
            #     plt.savefig("-1.png")
                plt.show()
            y = f(df_gulu.time)
            df_gulu = df_gulu.assign(flux=df_gulu['flux'].subtract(y),fit=df_gulu['fit'].add(y))
        #     plt.plot(df_gulu.time,df_gulu.flux)
        #     plt.show()
            np.savetxt(direc + '//' + fname.filename[-32:-5] + '_gulu.txt',df_gulu,fmt = '%f',header='Detrended')
        #    print(single_file)
#            overall.append(single_file)
#        np.savetxt(direc + '//' 'overall.txt',overall,fmt = '%s',header='Overall Details')
        return(True)
        
