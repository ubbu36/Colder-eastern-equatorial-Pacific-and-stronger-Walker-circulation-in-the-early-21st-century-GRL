#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:10:20 2022

@author: ullaheede
"""

model_names=['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','BCC-ESM1','CAMS-CSM1-0','CanESM5','CAS-ESM2-0','CESM2','CESM2-FV','CESM2-WACCM','CESM2-WACCM-FV2',\
             'CIESM','CMCC-CM2-SR5','CNRM-CM6','CNRM-CM6-HR','CNRM-ESM2-1','E3SM','FGOALS-f3-L','FGOALS-g3','GFDL-CM4','GFDL-ESM4','GISS-E2-1-G','GISS-E2-1-H','GISS-E2-2-H',\
             'HadGEM3-GC31-LL','HadGEM3-GC3-MM','INM-CM4-8','INM-CM5-0','IPSL-CM6A','KACE-1-0-G','MCM-UA-1-0','MIROC-ES2L','MIROC6','MPI-ESM-1-2-HAM','MPI-ESM1-2-LR',\
                 'MRI-ESM2','NESM3','NorCPM1','SAM0-UNICORN','TaiESM1','UKESM1-0-LL']
    
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import glob
import os
from pylab import *
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 16})

e1=180
e2=240

w1=130
w2=150

line=np.zeros(w2)

fig = figure(figsize=(15,6))
gs = gridspec.GridSpec(2, 6)
ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[0,2:4])
ax3 = plt.subplot(gs[0,4:6])


fig = gcf()
gs.tight_layout(fig,h_pad=3,w_pad=2)
axlist = [ax1,ax2,ax3]

plt.figtext(0.033, 0.975, 'a)')
plt.figtext(0.365, 0.975, 'b)')
plt.figtext(0.70, 0.975, 'c)')

mylist=xr.open_dataset('/Users/ullaheede_1/psl_historical_CESM2-FV2.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['psl'].sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
grad=west-east
grad=grad.rolling(year=10).mean()
grad_out_std_CESM=grad-grad.mean('year')
grad_out_mean_CESM=grad-grad.sel(year=slice(1950,1970)).mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/psl_historical_CESM2-FV2.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['psl'].sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
    grad=west-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_std_CESM=xr.concat([grad_out_std_CESM,grad_std], 'ens_member')
    grad_out_mean_CESM=xr.concat([grad_out_mean_CESM,grad_mean], 'ens_member')



mylist=xr.open_dataset('/Users/ullaheede_1/psl_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['psl'].sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
grad=west-east
grad=grad.rolling(year=10).mean()
grad_out_std=grad-grad.mean('year')
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/psl_historical_mean.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['psl'].sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
    grad=west-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')


grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member')

obs=xr.open_dataset('/Users/ullaheede_1/Downloads/slp.mon.mean.nc')
ts_obs=obs['slp']*100

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)

east_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(e1,e2)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(w1,w2)).mean('lat').mean('lon')
grad_obs=(west_obs-east_obs).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')
year_obs=grad_obs.year



line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-2*grad_std
te_std4=te+2*grad_std
#for x in range(0,len(grad_out)):
#    axlist[1].plot(year,grad_out[x],color='0.8',linewidth=1.0)
axlist[1].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[1].fill_between(year,te_std1,te_std2,facecolor='0.6')
te_obs=grad_obs

axlist[1].plot(year_obs,te_obs,'red',label='NOAA/NCEP reanalysis',linewidth=2.0)
axlist[1].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)
axlist[1].plot(year,grad_out_mean_CESM.mean('ens_member'),label='CESM2-FV2',color='green',linewidth=1.0)
#axlist[1].plot(year,grad_out_mean_CESM.isel(ens_member=1),label='CESM2-FV2',color='black',linewidth=1.0)
#axlist[1].plot(year,grad_out_mean_CESM.isel(ens_member=2),label='CESM2-FV2',color='purple',linewidth=1.0)

axlist[1].plot(year,line,'0.1')
axlist[1].set_xlim([1850, 2020])
#axlist[1].set_ylim([-1.5, 1])
axlist[1].set_title('Zonal SLP gradient anomalies',fontsize=15)
axlist[1].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[1].legend(loc="lower right")
axlist[1].set_ylabel('$\Delta$SLP (Pa)')
#axlist[1].legend(fontsize=10)
axlist[1].invert_yaxis()




#############################################################################

model_names=['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','BCC-ESM1','CanESM5','CESM2','CESM2-WACCM',\
           'CNRM-CM6','CNRM-ESM2-1','FGOALS-f3-L','GFDL-CM4','GFDL-ESM4','GISS-E2-1-G','GISS-E2-1-H',\
            'HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0','IPSL-CM6A','MCM-UA-1-0','MIROC-ES2L','MIROC6',\
           'MRI-ESM2','NESM3','UKESM1-0-LL']
    
mylist=xr.open_dataset('/Users/ullaheede_1/uas_historical_CESM2-FV2.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['uas']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(130,270)).mean('lat').mean('lon')
grad=west
grad=grad.rolling(year=10).mean()
grad_out_st_CESMd=grad-grad.mean('year')
grad_out_mean_CESM=grad-grad.sel(year=slice(1980,1985)).mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/uas_historical_CESM2-FV2.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['uas']
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat',skipna='True').mean('lon',skipna='True')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(130,270)).mean('lat',skipna='True').mean('lon',skipna='True')
    grad=west
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1980,1985)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_std_CESM=xr.concat([grad_out_std_CESM,grad_std], 'ens_member')
    grad_out_mean_CESM=xr.concat([grad_out_mean_CESM,grad_mean], 'ens_member')

mylist=xr.open_dataset('/Users/ullaheede_1/uas_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['uas']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(130,270)).mean('lat').mean('lon')
grad=west
grad=grad.rolling(year=10).mean()
grad_out_std=grad-grad.mean('year')
grad_out_mean=grad-grad.sel(year=slice(1980,1985)).mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/uas_historical_mean.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['uas']
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat',skipna='True').mean('lon',skipna='True')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(130,270)).mean('lat',skipna='True').mean('lon',skipna='True')
    grad=west
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1980,1985)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')



grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member')



obs=xr.open_dataset('/Users/ullaheede_1/Downloads/uwnd.mon.mean.nc')
ts_obs=obs['uwnd'].isel(level=0)


ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)


east_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(e1,e2)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(150,270)).mean('lat').mean('lon')
grad_obs=(west_obs).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1980,2020))-grad_obs.sel(year=slice(1980,1985)).mean('year')
year_obs=grad_obs.year

line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-2*grad_std
te_std4=te+2*grad_std
#for x in range(0,len(grad_out)):
#    axlist[1].plot(year,grad_out[x],color='0.8',linewidth=1.0)
axlist[2].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[2].fill_between(year,te_std1,te_std2,facecolor='0.6')
#axlist[2].plot(year,grad_out_mean_CESM.isel(ens_member=0),label='CESM2-FV2',color='green',linewidth=1.0)
#axlist[2].plot(year,grad_out_mean_CESM.isel(ens_member=1),label='CESM2-FV2',color='black',linewidth=1.0)
#axlist[2].plot(year,grad_out_mean_CESM.isel(ens_member=2),label='CESM2-FV2',color='purple',linewidth=1.0)
axlist[2].plot(year,grad_out_mean_CESM.mean('ens_member'),label='CESM2-FV2',color='green',linewidth=1.0)

te_obs=grad_obs
#axlist[2].plot(year,grad_CESM2,label='CESM2-FV2',color='green',linewidth=1.0)
axlist[2].plot(year_obs,te_obs,'red',label='NOAA/NCEP reanalysis',linewidth=2.0)
axlist[2].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)

axlist[2].plot(year,line,'0.1')
axlist[2].set_xlim([1850, 2020])
#axlist[2].set_ylim([-1.5, 1])
axlist[2].set_title('Zonal surface wind anom.',fontsize=15)
axlist[2].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[2].legend(loc="lower right")
axlist[2].set_ylabel('m/s')
axlist[2].invert_yaxis()
#axlist[2].legend(fontsize=10)



#%%

e1=200
e2=280

w1=80
w2=150


#############################################################################

model_names=['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','BCC-ESM1','CanESM5','CESM2','CESM2-WACCM',\
           'CNRM-CM6','CNRM-ESM2-1','FGOALS-f3-L','GFDL-CM4','GFDL-ESM4','GISS-E2-1-G','GISS-E2-1-H',\
            'HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0','IPSL-CM6A','MCM-UA-1-0','MIROC-ES2L','MIROC6',\
           'MRI-ESM2','NESM3','UKESM1-0-LL']
    
mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_CESM2-FV2.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['ts']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
grad=west-east
grad=grad.rolling(year=10).mean()
grad_out_std_CESM=grad-grad.mean('year')
grad_out_mean_CESM=grad-grad.sel(year=slice(1950,1970)).mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_CESM2-FV2.nc')
    ts_anom=mylist['ts'].isel(new_dim=x)
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
    grad=west-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_std_CESM=xr.concat([grad_out_std_CESM,grad_std], 'ens_member')
    grad_out_mean_CESM=xr.concat([grad_out_mean_CESM,grad_mean], 'ens_member')


mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['ts']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
grad=west-east
grad=grad.rolling(year=10).mean()
grad_out_std=grad-grad.mean('year')
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
    ts_anom=mylist['ts'].isel(new_dim=x)
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(e1,e2)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(w1,w2)).mean('lat').mean('lon')
    grad=west-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')


grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member')



obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
ts_obs=obs['sst']

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)

east_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(e1,e2)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(w1,w2)).mean('lat').mean('lon')
grad_obs=(west_obs-east_obs)
grad_obs=(grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')).rolling(year=10).mean()
year_obs=grad_obs.year

line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-2*grad_std
te_std4=te+2*grad_std
#for x in range(0,len(grad_out)):
#    axlist[1].plot(year,grad_out[x],color='0.8',linewidth=1.0)
axlist[0].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[0].fill_between(year,te_std1,te_std2,facecolor='0.6')

te_obs=grad_obs
#axlist[0].plot(year,grad_out_mean_CESM.isel(ens_member=0),label='CESM2-FV2',color='green',linewidth=1.0)
#axlist[0].plot(year,grad_out_mean_CESM.isel(ens_member=1),label='CESM2-FV2',color='black',linewidth=1.0)
#axlist[0].plot(year,grad_out_mean_CESM.isel(ens_member=2),label='CESM2-FV2',color='purple',linewidth=1.0)
axlist[0].plot(year,grad_out_mean_CESM.mean('ens_member'),label='CESM2-FV2',color='green',linewidth=1.0)

axlist[0].plot(year_obs,te_obs,'red',label='ersst v5',linewidth=2.0)

axlist[0].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)

axlist[0].plot(year,line,'0.1')
axlist[0].set_xlim([1850, 2020])
#axlist[0].set_ylim([-1.5, 1])
axlist[0].set_title('Zonal SST gradient anom.',fontsize=15)
axlist[0].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[0].legend(loc="lower right")
axlist[0].set_ylabel('$\Delta$SST ($^o$C)')
axlist[0].legend(fontsize=10)

fig.savefig('variable_comparison_supple.jpeg',dpi=1000, bbox_inches = "tight")
