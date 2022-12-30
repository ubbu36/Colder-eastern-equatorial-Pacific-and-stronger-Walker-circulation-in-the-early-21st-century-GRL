#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:08:10 2022

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

plt.rcParams.update({'font.size': 18})


model_names=['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','BCC-ESM1','CanESM5','CESM2','CESM2-WACCM',\
           'CNRM-CM6','CNRM-ESM2-1','FGOALS-f3-L','GFDL-CM4','GFDL-ESM4','GISS-E2-1-G','GISS-E2-1-H',\
            'HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0','IPSL-CM6A','MCM-UA-1-0','MIROC-ES2L','MIROC6',\
           'MRI-ESM2','NESM3','UKESM1-0-LL']
    
mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['ts']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(180,270)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(40,100)).mean('lat').mean('lon')
grad=west#-east
grad=grad.rolling(year=10).mean()
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
grad_out_std=grad-grad.mean('year')


for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['ts']
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(180,270)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(40,100)).mean('lat').mean('lon')
    grad=west#-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
 

grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member')


obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
ts_obs=obs['sst']

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)

east_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(180,270)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(40,100)).mean('lat').mean('lon')
grad_obs=(west_obs).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')
year_obs=grad_obs.year


fig = figure(figsize=(15,10))
gs = gridspec.GridSpec(2, 6)
ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[0,2:4])
ax3 = plt.subplot(gs[0,4:6])
ax4 = plt.subplot(gs[1,1:3])
ax5 = plt.subplot(gs[1,3:5])

fig = gcf()
gs.tight_layout(fig,h_pad=3,w_pad=2)
axlist = [ax1,ax2,ax3,ax4,ax5]

plt.figtext(0.033, 0.975, 'a)')
plt.figtext(0.365, 0.975, 'b)')
plt.figtext(0.70, 0.975, 'c)')
plt.figtext(0.20, 0.463, 'd)')
plt.figtext(0.54, 0.463, 'e)')

line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std

te_std3=te-grad_std*2
te_std4=te+grad_std*2

z_score=(grad_obs.sel(year=slice(1960,2015)).mean('year')-te.sel(year=slice(1960,2015)).mean('year'))/grad_std.sel(year=slice(1960,2015)).mean('year')

#for x in range(0,len(grad_out)):
#    axlist[0].plot(year,grad_out[x],color='grey',linewidth=1.0)
axlist[0].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[0].fill_between(year,te_std1,te_std2,facecolor='0.6')

te_obs=grad_obs
axlist[0].plot(year_obs,te_obs,'red',label='ersst v5',linewidth=2.0)
axlist[0].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)

axlist[0].plot(year,line,'0.1')
axlist[0].set_xlim([1850, 2020])
axlist[0].set_ylim([-1.5, 1])
axlist[0].set_title('Indian Ocean SST anom.')
axlist[0].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
axlist[0].legend(loc="lower right")
axlist[0].set_ylabel('$^o$C')

#%%

mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['ts']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(180,270)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(150,180)).mean('lat').mean('lon')
grad=west#-east
grad=grad.rolling(year=10).mean()
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
grad_out_std=grad-grad.mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['ts']
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(180,270)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(150,180)).mean('lat').mean('lon')
    grad=west#-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
 

grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member')

obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
ts_obs=obs['sst']

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)

east_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(180,270)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(150,180)).mean('lat').mean('lon')
grad_obs=(west_obs).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')
year_obs=grad_obs.year

line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-grad_std*2
te_std4=te+grad_std*2
#for x in range(0,len(grad_out)):
#    axlist[0].plot(year,grad_out[x],color='grey',linewidth=1.0)
axlist[3].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[3].fill_between(year,te_std1,te_std2,facecolor='0.6')
te_obs=grad_obs
#for x in range(0,len(grad_out)):
#    axlist[3].plot(year,grad_out[x],color='grey',linewidth=1.0)

axlist[3].plot(year_obs,te_obs,'red',label='ersst v5',linewidth=2.0)
axlist[3].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)
axlist[3].plot(year,line,'0.1')
axlist[3].set_xlim([1850, 2020])
axlist[3].set_ylim([-1.5, 1])
axlist[3].set_title('West Pacific SST anom.')
axlist[3].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[3].legend(loc="lower right")
axlist[3].set_ylabel('$^o$C')

#%%
mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)

ts_anom=ts_anom['ts']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(-5,5),lon=slice(180,270)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(310,340)).mean('lat').mean('lon')
grad=west#-east
grad=grad.rolling(year=10).mean()
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
grad_out_std=grad-grad.mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['ts'].sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(-5,5),lon=slice(180,270)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(310,340)).mean('lat').mean('lon')
    grad=west#-east
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
 

grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member')


obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
ts_obs=obs['sst']

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)

east_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(180,270)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(310,340)).mean('lat').mean('lon')
grad_obs=(west_obs).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')
year_obs=grad_obs.year

line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-grad_std*2
te_std4=te+grad_std*2
#for x in range(0,len(grad_out)):
#    axlist[0].plot(year,grad_out[x],color='grey',linewidth=1.0)
axlist[1].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[1].fill_between(year,te_std1,te_std2,facecolor='0.6')
te_obs=grad_obs
#for x in range(0,len(grad_out)):
#    axlist[1].plot(year,grad_out[x],color='grey',linewidth=1.0)

axlist[1].plot(year_obs,te_obs,'red',label='ersst v5',linewidth=2.0)
axlist[1].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)
axlist[1].plot(year,line,'0.1')
axlist[1].set_xlim([1850, 2020])
axlist[1].set_ylim([-1.5, 1])
axlist[1].set_title('Trop. Atlantic SST anom.')
axlist[1].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[1].legend(loc="lower right")
axlist[1].set_ylabel('$^o$C')

#%%
#filelist = glob.glob(os.path.join('/Volumes/Armor_CMIP6/', 'ts_historical*.nc'))
#filelist=sorted(filelist, key=lambda s: s.lower())
obs=xr.open_dataset('/Users/ullaheede_1/Downloads/ersst.v4.1854-2020.nc')

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )
regridder = xe.Regridder(obs, ds_out, 'bilinear')
mylist_obs_regrid = regridder(obs)



mask_ocean = 1 * np.ones((mylist_obs_regrid.dims['lat'], mylist_obs_regrid.dims['lon'])) * np.isfinite(mylist_obs_regrid.sst.isel(time=0))
test=mask_ocean.where(mask_ocean == 1)



mylist1=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
mylist=mylist1.isel(new_dim=0)

ts_anom=mylist['ts']
ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(20,65)).sel(lon=slice(180,280))
weights = np.cos(np.deg2rad(east.lat))
weights.name = "weights"
east_weighted = east.weighted(weights)
east_weighted_mean = east_weighted.mean(("lon", "lat"))

#west=ts_anom.sel(lat=slice(-5,5),lon=slice(180,280)).mean('lat').mean('lon')
grad=east_weighted_mean
grad=grad.rolling(year=10).mean()
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
grad_out_std=grad-grad.mean('year')

for x in range(1,len(mylist1.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
    mylist=mylist.isel(new_dim=x)
    ts_anom=mylist['ts']*test
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(20,65)).sel(lon=slice(180,280))
    weights = np.cos(np.deg2rad(east.lat))
    weights.name = "weights"
    east_weighted = east.weighted(weights)
    east_weighted_mean = east_weighted.mean(("lon", "lat"))

    grad=east_weighted_mean
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
 

grad_mean=grad_out_mean.mean('ens_member').isel(lev=0)

grad_std=grad_out_std.std('ens_member').isel(lev=0)


obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
ts_obs=obs['sst']

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)


east_obs=ts_obs_a.sel(lat=slice(65,20)).sel(lon=slice(120,240))
weights = np.cos(np.deg2rad(east_obs.lat))
weights.name = "weights"
east_obs_weighted = east_obs.weighted(weights)
east_obs_weighted_mean = east_obs_weighted.mean(("lon", "lat"))

west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(180,280)).mean('lat').mean('lon')
grad_obs=(east_obs_weighted_mean).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')
year_obs=grad_obs.year

line=np.zeros(165)
plt.figure(1)
year=np.arange(1850,2015,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-grad_std*2
te_std4=te+grad_std*2
#for x in range(0,len(grad_out)):
#    axlist[0].plot(year,grad_out[x],color='grey',linewidth=1.0)
axlist[2].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[2].fill_between(year,te_std1,te_std2,facecolor='0.6')
te_obs=grad_obs
#for x in range(0,len(grad_out)):
#    axlist[2].plot(year,grad_out[x],color='grey',linewidth=1.0)

axlist[2].plot(year_obs,te_obs,'red',label='ersst v5',linewidth=2.0)
axlist[2].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)
axlist[2].plot(year,line,'0.1')
axlist[2].set_xlim([1850, 2020])
#axlist[3].set_ylim([-1.5, 1])
axlist[2].set_title('North Pacific SST anom.')
axlist[2].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[2].legend(loc="lower right")
axlist[2].set_ylabel('$^o$C')

#%%
#filelist = glob.glob(os.path.join('/Volumes/Armor_CMIP6/', 'ts_historical*.nc'))
#filelist=sorted(filelist, key=lambda s: s.lower())

obs=xr.open_dataset('/Users/ullaheede_1/Downloads/ersst.v4.1854-2020.nc')

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )
regridder = xe.Regridder(obs, ds_out, 'bilinear')
mylist_obs_regrid = regridder(obs)



mask_ocean = 1 * np.ones((mylist_obs_regrid.dims['lat'], mylist_obs_regrid.dims['lon'])) * np.isfinite(mylist_obs_regrid.sst.isel(time=0))
test=mask_ocean.where(mask_ocean == 1).values



mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
ts_anom=mylist.isel(new_dim=0)
ts_anom=ts_anom['ts']*test

ts_anom=ts_anom.sel(year=slice(1850,2014))
east=ts_anom.sel(lat=slice(20,65),lon=slice(120,240)).mean('lat').mean('lon')
west=ts_anom.sel(lat=slice(-5,5),lon=slice(180,280)).mean('lat').mean('lon')
grad=west
grad=grad.rolling(year=10).mean()
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
grad_out_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
grad_out_std=grad-grad.mean('year')

for x in range(1,len(mylist.new_dim)):
#for x in range(11):
    mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_mean.nc')
    ts_anom=mylist.isel(new_dim=x)
    ts_anom=ts_anom['ts']*test
    ts_anom=ts_anom.sel(year=slice(1850,2014))
    east=ts_anom.sel(lat=slice(20,65),lon=slice(120,240)).mean('lat').mean('lon')
    west=ts_anom.sel(lat=slice(-5,5),lon=slice(180,280)).mean('lat').mean('lon')
    grad=west
    grad=grad.rolling(year=10).mean()
    grad_mean=grad-grad.sel(year=slice(1950,1970)).mean('year')
    grad_std=grad-grad.mean('year')
    grad_out_mean=xr.concat([grad_out_mean,grad_mean], 'ens_member')
    grad_out_std=xr.concat([grad_out_std,grad_std], 'ens_member')
 

grad_mean=grad_out_mean.mean('ens_member')

grad_std=grad_out_std.std('ens_member') 

obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
ts_obs=obs['sst']

ts_obs_a=ts_obs.groupby('time.year').mean('time',skipna=True)

east_obs=ts_obs_a.sel(lat=slice(20,65),lon=slice(120,240)).mean('lat').mean('lon')
west_obs=ts_obs_a.sel(lat=slice(5,-5),lon=slice(180,280)).mean('lat').mean('lon')
grad_obs=(west_obs).rolling(year=10).mean()
grad_obs=grad_obs.sel(year=slice(1950,2020))-grad_obs.sel(year=slice(1950,1970)).mean('year')
year_obs=grad_obs.year


line=np.zeros(165)
plt.figure(1)
#year=np.arange(2015,2100,1)
te=grad_mean
te_std1=te-grad_std
te_std2=te+grad_std
te_std3=te-grad_std*2
te_std4=te+grad_std*2
#for x in range(0,len(grad_out)):
#    axlist[0].plot(year,grad_out[x],color='grey',linewidth=1.0)
axlist[4].fill_between(year,te_std3,te_std4,facecolor='0.8')
axlist[4].fill_between(year,te_std1,te_std2,facecolor='0.6')
te_obs=grad_obs
#for x in range(0,len(grad_out)):
#    axlist[4].plot(year,grad_out[x],color='grey',linewidth=1.0)

axlist[4].plot(year_obs,te_obs,'red',label='ersst v5',linewidth=2.0)
#axlist[2].plot(year,grad_out_CESM,'blue',label='CESM2-FV2',linewidth=2.0)
axlist[4].plot(year,te,label='model mean',color='mediumblue',linewidth=2.0)
axlist[4].plot(year,line,'0.1')
axlist[4].set_xlim([1850, 2020])
axlist[4].set_ylim([-1.5, 1])
axlist[4].set_title('East Pacific SST anom')
axlist[4].set_xlabel('years')
#axlist[2].set_ylabel('$\Delta$ K')
#axlist[4].legend(loc="lower right")
axlist[4].set_ylabel('$^o$C')

