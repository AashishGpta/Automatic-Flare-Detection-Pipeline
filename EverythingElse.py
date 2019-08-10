import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
#import os
import pandas as pd

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

from scipy.signal import wiener

import statsmodels.api as sm

from astropy.stats import median_absolute_deviation as mad

def flare_template_d(t,td):
    c = maxval*np.exp(-(t-t[0])/td)
# def flare_fit_d(t,t0,td,b):
#     c = max(ff)*np.exp(-(t-t0)/td) + b
    return c

def flare_fit(segt,segf,sege):
    popt, pcov = curve_fit(flare_template_d, segt, segf,[(segt[-1]-segt[0])/np.e],sigma = sege,absolute_sigma=True, bounds=(0, 5))#,maxfev = 10000)
#     plt.plot(np.flipud(segt),flare_fit_d(np.array(segt),*popt))#,'r--')
    return(popt[0],np.sqrt(pcov[0][0]))


# # FLATW'RM Files
# for fname in fits_files:
#     df_temp = pd.read_csv(direc + '//' + fname,sep = ' ',skiprows = 1,header = None)
#     df_temp.columns = ['time','flux','err','original1','original2','cos','phase','fit']
#     df_temp = df_temp.sort_values(by = 'time')
#     np.savetxt(direc + '//' +"FLATWRM_" + fname,np.array(df_temp.iloc[:,[0,3]],dtype= float),fmt = '%f')
def analysis(kic,df_big,row):
    direc = str(kic)
#    df_big = pd.read_csv(direc + '//' + direc + '_df_big.txt',sep = ' ',header = None)
#    df_big.columns = ['time','flux','err','original1','original2','straight','fit','cos','phase']
#    df_big = df_big.sort_values(by = 'time')
#    df_big.index = range(len(df_big))
    
    med = np.median(df_big.flux)
    # temp_thresh = 2*float(np.std(df_big.flux))
    temp_thresh = 7*float(mad(df_big.flux))
    df_temp = df_big[(df_big.flux < med + temp_thresh) & (df_big.flux > med -temp_thresh)]
    f = interp1d(df_temp.time,sm.tsa.filters.hpfilter(df_temp.flux, lamb=1e7)[1],kind='linear',fill_value = 'extrapolate')
    y = f(df_big.time)
    # plt.plot(df_big.time,df_big.flux)
    # plt.plot(df_big.time,y)
    # plt.show()
    df_big = df_big.assign(flux=df_big.flux.subtract(y),fit=df_big.fit.add(y) )
    #df_big = df_big.assign(flux=df_big['flux'].subtract(y) )
    
    df_gulu=df_big.loc[:,'time':'err']
    df_exp = df_big.iloc[:,[0,4,2]]
    
    data_details = np.load(direc + '//' + direc + '_data_details.npy')
    period = data_details[0]
    
    #if True:
    print("Filtering Data")
    
    #     df_filterred = pd.DataFrame([df_gulu.time,df_gulu.flux,df_gulu.err]).T#medfilt(df_exp.flux,3)]).T
    #     df_filterred.columns =['time','flux','err']
    df_filterred = df_gulu.copy()   
    #     plt.plot(df_filterred.time,df_filterred.flux,label = 'Original')
    
    noise1 = np.median(df_gulu.err)
    noise = 1.5*noise1
    minimafilt = detect_peaks(df_filterred.flux,show = False,valley=True)
    
    for k in range(len(minimafilt)-1):
    #         bin1 = tuple(minimafilt[k],minimafilt[k+1]+1)
    #         if max(df_filterred.flux[bin1[0]:bin1[-1]]) - min(df_filterred.flux[minimafilt[k]],df_filterred.flux[minimafilt[k+1]])<noise:
    #                 df_filterred.flux[bin1[0]:bin1[-1]] = np.linspace(df_filterred.flux[bin1[0]],df_filterred.flux[bin1[-1]],len(bin1))
        #bin1 = list(range(minimafilt[k],minimafilt[k+1]+1))
        a,b = minimafilt[k],minimafilt[k+1]
        m = np.max(df_filterred.flux[a+1:b])
        if m - max(df_filterred.flux[a],df_filterred.flux[b])<noise:
                df_filterred.flux[a+1:b] = np.full(b-a-1,sum(df_gulu.flux[a+1:b])/(b-a-1))   #np.linspace(df_filterred.flux[a],df_filterred.flux[b],b-a+1)[1:-1]
    #                 df_filterred.flux[a:b] = np.linspace(df_filterred.flux[a],df_filterred.flux[b-1],b-a)
    
    # if True:    
    #     df_filterred.flux = wiener(df_filterred.flux,3,5*noise1)
    df_filterred.flux = wiener(df_filterred.flux,2)
        
    #     plt.plot(df_gulu.time,df_gulu.flux,label = 'Original')
    #     plt.plot(df_filterred.time,df_filterred.flux,label = 'Filtered')
    #     plt.legend()
    #     plt.xlabel("Time (d)")
    #     plt.ylabel("Flux")
    # #     plt.savefig("xzc.png")
    #     plt.show()
    
    df_133 = df_filterred.copy()    
    df_filterred = df_gulu.copy()
    
    # if True:
    #     df_.index = range(len(df_final))
    #     df_133 = df_final.copy()
    #     mask = df_final.flux<-1.5*np.std(df_final.flux)
    #     df_133.loc[mask, 'flux'] = -1.5*np.std(df_final.flux)
    #     df_133.index = range(len(df_133))
    #     thresh = np.median(df_133.flux)# + noise/2
    med= np.median(df_133.flux) + noise1
    mask = df_133.flux<med
    df_133.loc[mask, 'flux'] = med
    #     df_133.flux = wiener(df_133.flux,3)
    
    
    mask = df_gulu.flux<med  
    df_gulu.loc[mask, 'flux'] = med
    
    #     df_gulu.loc[mask, 'flux'] = thresh
    
    
    times = np.array(df_exp.time)
    dt = times[1:]-times[:-1]
    gap = np.append([-1],np.where(dt >= 0.2))
    gap = np.concatenate((gap,gap + 1))
    
    
    df_gulu.flux.iloc[gap] = med
    
    # df_1 = df_filterred.copy()
    # df_2 = df_gulu.copy()
    # df_3 = df_133.copy()
    
    # df_filterred = df_1.copy()
    # df_gulu = df_2.copy()
    # df_133 = df_3.copy()
    
    # if True:
    print("Detecting Flares")
    
    minimas = detect_peaks(df_133.flux,show = False,valley=True)
    #b1 = np.median(df_133.flux[minimas])
    b1 = med #np.mean(df_133.flux[minimas])
    b2 = noise1
    minimas332 = []
    minimas332.append(minimas[0])
    for i in range(1,len(minimas)-1):
        if (df_133.flux[minimas[i]] < b1 + 1*b2):
            minimas332.append(minimas[i])
        elif (df_133.flux[minimas[i]] < b1 + 3*b2):
            if (max(df_133.flux[minimas[i]:minimas[i+1]+1]) - df_133.flux[minimas[i]] < noise) or (max(df_133.flux[minimas[i-1]:minimas[i]+1]) - df_133.flux[minimas[i]] < noise):
                continue
            minimas332.append(minimas[i])
    
    minimas332.append(minimas[-1])
    #     plt.plot(df_133.time,df_133.flux)
    #     plt.plot(df_133.time,[b1+b2]*len(df_133))
    #     plt.plot(df_133.time[minimas332],df_133.flux[minimas332],'.')
    #     plt.show()
    
    # if True:
    flares = []
    bins = []
    thresh = np.median(df_133.flux) + noise1
    for k in range(len(minimas332)-1):
        if minimas332[k+1] - minimas332[k] > 2:
            if max(df_133.flux[minimas332[k]:minimas332[k+1]+1]) > thresh:
                
                bins.append(list(range(minimas332[k],minimas332[k+1]+1)))
    
    # if True:
    n = 15
    thresh_prev = np.max(df_filterred['flux'])
    for j in range(n):
        k=0
        # print("Identifying Flares")
        # Without Correction thing
    #         mean,std = weighted_stats(df_filterred.flux,df_filterred.err**-1)
        mean,std = np.median(df_filterred.flux),np.std(df_filterred.flux)
        thresh =max(2*noise1 + 1.5*std,2.5*std) + mean # Can  also use noise + 1.5*std
        thresh2 = max(noise1 + 1*std,1.5*std) + mean
    #         thresh =max(noise/2 + 2*np.std(df_filterred.flux),2.5*np.std(df_filterred.flux)) + np.median(df_filterred.flux)#np.mean(df_filterred['flux'])+3*np.std(df_filterred['flux'])#,1.5*noise)#+np.median(df_filterred['flux'])
    #         thresh =max(2.5*np.std(df_filterred['flux']),1.5*noise)#+np.median(df_filterred['flux'])
    #         thresh =3*np.std(df_133['flux'])
        if thresh_prev-thresh < noise1/25:
    #         if thresh_prev-thresh < 0.1 or 1.5*noise > thresh:
    #             print(thresh_prev,thresh,noise,1.1*noise,1.5*noise)
            break
        thresh_prev = thresh
    #     plt.plot(df_133.time,df_133.flux)
    #     plt.plot(df_133.time,[thresh]*len(df_133.flux))
    #     plt.show()
    #         print(thresh)#,np.mean(df_133['flux']))
        #thresh = 2*np.std(df_132.flux[minimaf])#+np.mean(df_132.flux[minimaf])
        #thresh = threshfunc(np.array(df_132.flux[minimaf2]))
    #     for k in range(len(bins)):
    #         i = bins[k]
    #         if np.max(df_133.flux[i]) > thresh:
    #             flare_time.append([df_133.time[i[0]],df_133.time[i[-1]]])
    #             bins.remove(i)
    #             df_133.drop(i)
    #             k = k-1
        while k<len(bins):
            i = bins[k]
    #             if np.amax(np.array(df_133.flux[i][df_133.flux[i]<max(df_133.flux[i])]),initial=0) > thresh:
    #             if np.sort(df_133.flux[i])[-2] > thresh:
            amax = df_133.flux[i[1:-1]].idxmax()
            if df_133.flux[amax] > thresh:
    #                 if max(np.concatenate((df_133.flux[i[1]:amax],df_133.flux[amax+1:i[-1]]))) > thresh2:
                if max(df_133.flux[amax-1],df_133.flux[amax+1]) > thresh2:
                    if (df_133.time[i[-1]]-df_133.time[i[0]])< period:
    #                 if (df_133.time[i[-1]]-df_133.time[i[0]])< (1/freq_main):
    #                     print(float(df_gulu.flux[df_gulu.time == df_133.time[np.argmax(df_133.flux[i[1:-1]])]]),max(float(df_gulu.flux[df_gulu.time == df_133.time[i[-1]]]),float(df_gulu.flux[df_gulu.time == df_133.time[i[-1]]])))
    #                         flare_time.append([df_133.time[i[0]],df_133.time[i[-1]]])
                        flares.append(i)
    #                     time_max = time_max + [df_133.time[np.argmax(df_133.flux[i[1:-1]])]]
    #                     if abs(df_133.time[np.argmax(df_133.flux[i[1:-1]])]-1572.26) < 0.01:
    #                         break
                        bins.remove(i)
    #                         df_filterred.drop(i[1:-1],inplace=True)
                        #print(i)
                        k = k - 1
                    else:
                        bins.remove(i)
                        k = k - 1
    
                else:
                    bins.remove(i)
                    k = k - 1
            k = k + 1
        df_filterred = df_filterred[(df_filterred.flux<thresh) & (df_filterred.flux>-thresh)]
    #         len(flares)/(times[-1]-times[0]-np.sum(times[gap[int(len(gap)/2)+1:]]-times[gap[1:int(len(gap)/2)]]))
    #         df_133 = df_133[(df_133.flux<thresh) & (df_133.flux>-thresh)]
        #if j>10:
        print(j+1,thresh,len(flares),len(bins))
    print(len(flares))

    df_big['clipped'] = df_gulu.flux
    df_big['filterred'] = df_133.flux
    df_big.to_pickle(direc + '//df_big.pkl')
    np.save(direc + '//flare_index.npy',flares)
    
    flare_freq = len(flares)/(times[-1]-times[0]-np.sum(times[gap[int(len(gap)/2)+1:]]-times[gap[1:int(len(gap)/2)]]))
    
    fnu = 3033.1*1e-23*(10**(-0.4*(float(row.kepmag))))
    
    lamda = 5836.3
    dlamda = 10827-3321
    
    f = fnu*2.998e+18*dlamda/lamda**2
    
    l = (4*np.pi*(float(row.r_est)*3.086e+18)**2)*f
    
    print("Source luminosity:",l)
    print("Analysing flares")
    
    obj_details = np.concatenate([[kic,std,noise1,l,flare_freq],data_details,row[['radius','mass','teff']]])

    np.save(direc + '//Object_details.npy',obj_details)
    
    global maxval
            
    if len(flares) != 0:
        details = []
        flare_thresh = max(2*std,2*noise1)
        # flares=np.array([np.array(xi) for xi in flares])
        for i in range(len(flares)):
        # for i in range(9,10):
        # for i in [2260, 2492, 2498, 2499, 3674, 3845, 3846, 3847, 4759, 5650, 5651]:
        #     i = int(i)
            single = [i]
            flare = df_big.iloc[flares[i]]
            ff = np.array(flare.flux)
            minima_flare = detect_peaks(ff,show = False,valley=True,edge = 'rising')
    
        # if True: 
            if len(minima_flare)>1:
                if ff[0]<ff[1]:
                    minima_flare = np.concatenate(([0],minima_flare))
                if ff[-1]<ff[-2]:
                    minima_flare = np.concatenate((minima_flare,[len(ff)-1]))
            elif len(minima_flare)==1:
                if minima_flare[0] == 1:
                    minima_flare = np.concatenate((minima_flare,[len(ff)-1]))
                elif minima_flare[0] == len(ff)-2:
                    minima_flare = np.concatenate(([0],minima_flare))   
                else:
                    minima_flare = [0,len(ff)-1]
            else:
                minima_flare = [0,len(ff)-1]
    
    
            low_f = ff[minima_flare]
    
            starti = minima_flare[0]       # Redundant. 
            endi = minima_flare[-1] + 1
        #     print(minima_flare)
            ff = ff[starti:endi]
            tf = np.array(flare.time)[starti:endi]
            ef = np.array(flare.err)[starti:endi]
    
        # if True:
        #     ff = np.array(ff)
            a = detect_peaks(ff,show = False,valley=False)
            a= a[a<minima_flare[-1]]
            high_f = ff[a]
            if len(high_f) == 0:
        #        single = [np.nan]*len(details[-1])  #Replace by actual length of single
                continue
            try:
                single.append(np.count_nonzero(np.minimum(high_f-low_f[:-1],high_f-low_f[1:])>flare_thresh)) #No. of part flares
        #         if single[-1] == 0:    
        #             print(i)
            except Exception as e:
        #         print(e)
                print(str(i)+" Exception ocurred 1",e)
                single.append(np.nan) #No. of part flares
    
        #     ff = ff-np.linspace(ff[0],ff[-1],len(ff))#np.linspace(min(ff[minima_flare[0]],ff[0]),min(ff[minima_flare[-1]],ff[-1]),len(ff))
            fmax = np.argmax(ff)
            maxval = ff[fmax]
            single.append(maxval)   # Maximum Relative Flux
            if single[1] == 0:
                if single[2] > max(ff[0],ff[-1]) + noise:
                    single[1] = 1
        #      else: remove i from flares..... To be added
    
    
        #     single.append(ef[fmax]+max(ef[0],ef[-1]))    # Maximum Relative Flux Error
        #     single.append(tf[fmax])      # Maximum Relative Flux Time
        #     single.append(tf[fmax+1]-tf[fmax-1])     # Maximum Relative Flux Time Error
        #     single.append(tf[-1]-tf[0])      # Baseline Time
        #     single.append(tf[1]-tf[0]+tf[-2]-tf[-1])     # Baseline Time Error
            single.extend([ef[fmax],tf[fmax],tf[fmax+1]-tf[fmax-1],tf[-1]-tf[0],tf[1]-tf[0]+tf[-1]-tf[-2]])
            segt = tf[fmax]-tf[:fmax+1] +tf[0]
            segt = np.flipud(segt)
            segf = ff[:fmax+1]
            segf = np.flipud(segf)
            sege = ef[:fmax+1]
            sege = np.flipud(sege)
            if len(segt)>2:
                try:
                    tr,tr_err = flare_fit(segt,segf,sege)
                    single.extend([tr,tr_err])
            #         print(segt[-1]-segt[0],tr,tr_err)
                except Exception as e:
                    print(str(i)+" Exception ocurred 2",e)
                    single.extend([np.nan,np.nan])
            #         plt.show()
                    continue
                fit = np.flipud(flare_template_d(np.array(segt),tr))[:-1]
            else:
                single.extend([np.nan,np.nan])
                fit = np.full(len(segt),np.nan)
            segt = tf[fmax:]
            segf = ff[fmax:]#-ff[:5]+ff[0]
            sege = ef[fmax:]#-ff[:5]+ff[0]
            if len(segt)>2:
                try:
                    td,td_err = flare_fit(segt,segf,sege)
                    single.extend([td,td_err])
            #         print(segt[-1]-segt[0],td,td_err)
                except Exception as e:
                    print(str(i)+" Exception ocurred 3",e)
                    single.extend([np.nan,np.nan])
            #         plt.show()
                    continue
                fit = np.concatenate([fit,flare_template_d(np.array(segt),td)])
            else:
                single.extend([np.nan,np.nan])
                fit = np.concatenate([fit,np.full(len(segt),np.nan)])
            bmax = flare.original1.idxmax()
            single.extend([flare.original1[bmax],flare.err[bmax],np.median(flare.fit),flare.fit[bmax],flare.phase[bmax]])
        #     f0 = np.median(flare.fit)
            single.append(np.trapz(flare.flux/flare.fit,flare.time*86400)*l)   
            if False:
                plt.errorbar(flare.time,flare.flux,yerr = flare.err)
                plt.plot(tf,fit)
            #     plt.savefig("Flare-Fit.png")
                plt.show()
                print(single)
            details.append(single)
        #     plt.plot(df_big.time[flares[i]],df_big.original1[flares[i]]/df_big.fit[flares[i]])
        #     plt.show()
        details_np = np.array(details)
        details_np = details_np[details_np[:,4].argsort()]
        details_np = np.c_[details_np,np.append(details_np[:,4][1:]-details_np[:,4][:-1],np.nan)]
        np.save(direc + '//flare_details.npy',details_np)
        np.savetxt(direc + '//flare_details.txt',details_np,header = 'S.No., No. of subflares, Maximum relative flux value, error, Time of maximum value, error, Flare duration, error, Rise time, error, Decay time, error, Original flux value, error, Fit value 1, Fit value 2, Phase value, Energy, Time for next flare',fmt = '%5f')
        # Percentage of complex flares = (len(details_np[details_np[:,1]>1])/len(details_np))*100
        
        return(len(details_np))
    
