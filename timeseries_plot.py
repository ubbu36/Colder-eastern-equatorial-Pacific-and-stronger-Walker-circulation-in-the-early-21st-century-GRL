#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:05:06 2022

@author: ullaheede
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import glob as glob
import os
import scipy
#import xrscipy.signal as dsp
from pylab import *
import matplotlib.gridspec as gridspec

#%% SST
plt.rcParams.update({'font.size': 32})
k=10

sst_low=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')

#sst_low=sst_low['sst'].groupby('time.month')-sst_low['sst'].isel(time=slice(2014-20*12,2014)).groupby('time.month').mean('time',skipna=True)


sst_l_east=sst_low.sel(lon=slice(200,280),lat=slice(5,-5)).mean('lat',skipna=True).mean('lon',skipna=True)

sst_l_west=sst_low.sel(lon=slice(80,150),lat=slice(5,-5)).mean('lat',skipna=True).mean('lon',skipna=True)

grad_l=sst_l_west.rolling(time=k*12, center=True).mean().dropna("time")-sst_l_east.rolling(time=k*12, center=True).mean().dropna("time")
#%%
fig = figure(figsize=(51,23))
gs = gridspec.GridSpec(2, 4)
ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0, 1:2])
ax3 = plt.subplot(gs[0, 2:3])
ax4 = plt.subplot(gs[0, 3:4])

ax5 = plt.subplot(gs[1, 0:1])
ax6 = plt.subplot(gs[1, 1:2])
ax7 = plt.subplot(gs[1, 2:3])

ax8 = plt.subplot(gs[1, 3:4])
#ax4 = plt.subplot(gs[1, 0:3])

fig = gcf()
gs.tight_layout(fig,h_pad=7,w_pad=3.5)
ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

plt.figtext(0.025, 0.98, 'a)')
plt.figtext(0.28, 0.98, 'b)')
plt.figtext(0.535, 0.98, 'c)')
plt.figtext(0.787, 0.98, 'd)')

plt.figtext(0.025, 0.435, 'e)')
plt.figtext(0.28, 0.435, 'f)')
plt.figtext(0.535, 0.435, 'g)')
plt.figtext(0.787, 0.435, 'h)')

ax[0].set_title('Pacific SST gradient',fontsize=40)
ax[0].plot(sst_l_west.rolling(time=k*12, center=True).mean().dropna("time").time,grad_l['sst'],linewidth=7.0)
ax[0].set_ylabel('$\Delta$SST ($^o$C)')
ax[0].set_xlabel('years')
ax[0].set_ylim(2.6,3.6)
ax[0].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[0], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[0].set_xticklabels(['1980','1990','2000','2010','2020'])
#%% SSH
#k=1
ssh_background=xr.open_dataset('/Users/ullaheede_1/Downloads/zos_AVISO_L4_199210-201012 _1.nc')
ssh_background=ssh_background.mean('time')
ssh_background=ssh_background.rename_vars({'zos':'SLA'})

ssh_low=xr.open_dataset('/Users/ullaheede_1/Downloads/ssh_monthlyMean.nc')

ssh_l_east=ssh_low.sel(Longitude=slice(200,280),Latitude=slice(-5,5)).mean('Latitude',skipna=True).mean('Longitude',skipna=True) \
    + ssh_background.sel(lat=slice(-5,5)).sel(lon=slice(200,280)).mean('lat',skipna=True).mean('lon',skipna=True)

ssh_l_west=ssh_low.sel(Longitude=slice(130,180),Latitude=slice(-5,5)).mean('Latitude',skipna=True).mean('Longitude',skipna=True)  \
    + ssh_background.sel(lat=slice(-5,5)).sel(lon=slice(120,150)).mean('lat',skipna=True).mean('lon',skipna=True)

grad_l=ssh_l_west.rolling(Time=k*12, center=True).mean().dropna("Time")-ssh_l_east.rolling(Time=k*12, center=True).mean().dropna("Time")

ax[1].set_title('Pacific SHH gradient',fontsize=40)
ax[1].plot(ssh_l_west.rolling(Time=k*12, center=True).mean().dropna("Time").Time,grad_l['SLA'],linewidth=7.0)
ax[1].set_ylabel('$\Delta$ SLH (m)')
ax[1].set_xlabel('years')
#ax[0].set_xlim(695720.51+70*373, 739182.9566666667)
ax[1].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])

plt.setp(ax[1], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[1].set_xticklabels(['1980','1990','2000','2010','2020'])
#%% WINS
#k=10
olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/uwnd.mon.mean.nc')
olr_west=olr_low['uwnd'].isel(level=0).sel(lon=slice(150,280),lat=slice(5,-5)).mean('lat',skipna=True).mean('lon',skipna=True)

ax[2].set_title('Pacific zonal surface winds',fontsize=40)
ax[2].plot(olr_west.rolling(time=k*12, center=True).mean().dropna("time").time,olr_west.rolling(time=k*12, center=True).mean().dropna("time"),linewidth=7.0)
ax[2].set_ylabel('m/s')
ax[2].set_xlabel('years')
ax[2].invert_yaxis()
ax[2].set_ylim(-3,-3.8)

ax[2].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[2], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[2].set_xticklabels(['1980','1990','2000','2010','2020'])
#%%
#k=8
olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/slp.mon.mean.nc')
olr_west=olr_low['slp'].sel(lon=slice(130,150),lat=slice(5,-5)).mean('lat',skipna=True).mean('lon',skipna=True)
olr_east=olr_low['slp'].sel(lon=slice(180,240),lat=slice(5,-5)).mean('lat',skipna=True).mean('lon',skipna=True)

grad=olr_west.rolling(time=k*12, center=True).mean().dropna("time")-olr_east.rolling(time=k*12, center=True).mean().dropna("time")
ax[3].set_title('Pacific SLP gradient',fontsize=40)
ax[3].plot(olr_west.rolling(time=k*12, center=True).mean().dropna("time").time,grad,linewidth=7.0)
ax[3].set_ylabel('$\Delta$SLP (hPa)')
ax[3].set_xlabel('years')
ax[3].invert_yaxis()

ax[3].set_xlim(695720.51+70*373, 739182.9566666667)
#ax[0].set_xlim(695720.51+70*373, 739182.9566666667)
ax[3].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[3], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[3].set_xticklabels(['1980','1990','2000','2010','2020'])

#%%
#k=8
olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/precip.mon.mean.nc')
olr_west=olr_low['precip'].sel(lon=slice(130,150),lat=slice(10,-10)).mean('lat',skipna=True).mean('lon',skipna=True)
olr_east=olr_low['precip'].sel(lon=slice(180,280),lat=slice(10,-10)).mean('lat',skipna=True).mean('lon',skipna=True)

grad=olr_west.rolling(time=k*12, center=True).mean().dropna("time")-olr_east.rolling(time=k*12, center=True).mean().dropna("time")
ax[4].set_title('Zonal precipitation gradient',fontsize=40)
ax[4].plot(olr_west.rolling(time=k*12, center=True).mean().dropna("time").time,grad,linewidth=7.0)
ax[4].set_ylabel('$\Delta Rainfall$ (mm/day)')
ax[4].set_xlabel('years')

ax[4].set_xlim(695720.51+70*373, 739182.9566666667)
#ax[0].set_xlim(695720.51+70*373, 739182.9566666667)


ax[4].set_xlim(695720.51+70*373, 739182.9566666667)
#ax[0].set_xlim(695720.51+70*373, 739182.9566666667)
ax[4].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[4], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[4].set_xticklabels(['1980','1990','2000','2010','2020'])




#%%OSCAR
oscar=xr.open_dataset('/Users/ullaheede_1/Downloads/oscar_monthlyMean.nc')
oscar=oscar.isel(time=slice(0,len(oscar.time)-30))
#k=8

oscar_eq=oscar['u'].isel(depth=0).sel(longitude=slice(150,280),latitude=slice(3,-3)).mean('latitude',skipna=True).mean('longitude',skipna=True)
oscar_trend=oscar.isel(time=slice(380-5*12,380)).mean('time')-oscar.isel(time=slice(0,5*12)).mean('time')


#ax[5].plot(oscar_eq.rolling(time=1*12, center=True).mean().dropna("time").time,oscar_eq.rolling(time=12*1, center=True).mean().dropna("time"))
ax[5].plot(oscar_eq.rolling(time=k*12, center=True).mean().dropna("time").time,oscar_eq.rolling(time=12*k, center=True).mean().dropna("time"),linewidth=7.0)
ax[5].set_title('Zonal ocean surface current',fontsize=40)
ax[5].set_ylabel('m/s')
ax[5].set_xlabel('years')
ax[5].invert_yaxis()
#ax[5].set_xlim(695720.51+70*373, 739182.9566666667)
#ax[0].set_xlim(695720.51+70*373, 739182.9566666667)
ax[5].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[5], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[5].set_xticklabels(['1980','1990','2000','2010','2020'])

#%% omega
#k=8
olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/omega.mon.mean.nc')
olr_west=olr_low['omega'].sel(level=500).sel(lon=slice(130,150),lat=slice(10,-10)).mean('lat',skipna=True).mean('lon',skipna=True)
olr_east=olr_low['omega'].sel(level=500).sel(lon=slice(180,280),lat=slice(10,-10)).mean('lat',skipna=True).mean('lon',skipna=True)

grad=olr_west.rolling(time=k*12, center=True).mean().dropna("time")-olr_east.rolling(time=k*12, center=True).mean().dropna("time")
ax[6].set_title('West Pacific Omega 500 mb',fontsize=40)
ax[6].plot(olr_west.rolling(time=k*12, center=True).mean().dropna("time").time,olr_west.rolling(time=k*12, center=True).mean().dropna("time"),linewidth=7.0)
ax[6].set_ylabel('N/m$^2$ s')
ax[6].set_xlabel('years')
ax[6].invert_yaxis()
ax[6].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[6], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[6].set_xticklabels(['1980','1990','2000','2010','2020'])

#%% omega
#k=8
olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/olr.mon.mean.nc')
olr_west=olr_low['olr'].sel(lon=slice(120,150),lat=slice(10,-10)).mean('lat',skipna=True).mean('lon',skipna=True)
olr_east=olr_low['olr'].sel(lon=slice(180,240),lat=slice(10,-10)).mean('lat',skipna=True).mean('lon',skipna=True)

grad=olr_west.rolling(time=k*12, center=True).mean().dropna("time")-olr_east.rolling(time=k*12, center=True).mean().dropna("time")
ax[7].set_title('Central Pacific OLR',fontsize=40)
ax[7].plot(olr_east.rolling(time=k*12, center=True).mean().dropna("time").time,olr_east.rolling(time=k*12, center=True).mean().dropna("time"),linewidth=7.0)
ax[7].set_ylabel('W/m$^2$')
ax[7].set_xlabel('years')

ax[7].set_xlim([datetime.date(1980, 1, 1)], [datetime.date(2020, 1, 1)])
plt.setp(ax[7], xticks=([datetime.date(1980, 1, 1)],[datetime.date(1990, 1, 1)],[datetime.date(2000, 1, 1)],[datetime.date(2010, 1, 1)],[datetime.date(2020, 1, 1)]))
ax[7].set_xticklabels(['1980','1990','2000','2010','2020'])

fig.savefig('Heede_Fig1a.jpeg',dpi=400, bbox_inches = "tight")