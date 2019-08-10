# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:38:47 2018

@author: Aashish gupta
"""

from detrend import detrend
from phase import phase
from EverythingElse import analysis
from pandas import read_pickle 

gaia_df = read_pickle('gaia_df.pkl')

print("Enter Kepler IDs of the objects you are interested in.\nEnter 'stop' when done.") #Can set alternate methods for this input.
try: 
    kepler_ids = [] 
      
    while True: 
        kepler_ids.append(int(input())) 
          
except: 
    print('Kepler IDs: ',kepler_ids) 
   
for kic in kepler_ids: 
    
    row = gaia_df[gaia_df.kepid==kic]
    if len(row)>1:
        row = row.loc[row.kepler_gaia_ang_dist.idxmin()]
    elif len(row)==0:
        print("Not Found in gaia") # Think Something Better
        continue
    else:
        row = row.iloc[0]
    
    if detrend(kic):
        data = phase(kic)
        flare_no = analysis(kic,data,row)
    
    print("No. of flares found:",flare_no)
    
    
