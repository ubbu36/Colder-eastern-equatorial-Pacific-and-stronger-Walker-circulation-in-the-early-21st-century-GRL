#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:45:39 2022

@author: ullaheede
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
from mpl_toolkits.basemap import Basemap
import cartopy.feature as cfeature
from matplotlib import colorbar, colors
from matplotlib.cm import get_cmap
from matplotlib import colorbar, colors
import glob as glob
import os
import cmocean
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from scipy import stats
import matplotlib.gridspec as gridspec
from pylab import *
from eofs.xarray import Eof
import xrscipy.signal as dsp
obs=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-88, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )
obs=obs.mean('time').drop_vars('time_bnds').squeeze()

regridder = xe.Regridder(obs, ds_out, 'bilinear')
mylist_obs_regrid = regridder(obs)


mask_ocean = 1 * np.ones((mylist_obs_regrid.dims['lat'], mylist_obs_regrid.dims['lon'])) * np.isfinite(mylist_obs_regrid.sst)
test=mask_ocean.where(mask_ocean == 1)

mylist=xr.open_dataset('/Users/ullaheede_1/ts_historical_CESM2-FV2.nc')
sst_main=mylist.mean('new_dim')*test
sst_main=sst_main.rename(year="time")
sst=sst_main.rolling(time=5, center=True).mean().sel(time=slice('1914','2009'))

sst_low=(sst['ts']).sel(lat=slice(20,65),lon=slice(120,260))


sst_low=sst_low.assign_coords(time=range(0,len(sst_low.time)))


#sst_low= dsp.bandpass(sst_low,1/(40*12),1/(13*12),dim='time')
#sst_low= dsp.lowpass(sst_low,1/(1*12),dim='time')

sst_low=sst_low.transpose("time", "lon", "lat")

solver = Eof(sst_low)

eofs = solver.eofs(neofs=2)
pcs = solver.pcs(npcs=2)
eofs=eofs.transpose("lat", "lon",'mode')

nx=len(sst.lon)
ny=len(sst.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = pcs.sel(mode=1).values
y = sst['ts'].values

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray


sst_main['trends'] = (('lat', 'lon'), trends*10)
sst_main['trends_std_err'] = (('lat', 'lon'), trends_std_err)
sst_main['p_values'] = (('lat', 'lon'), p_values)

#%%
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(15, 6),subplot_kw={'projection': ccrs.PlateCarree(180)})

plt.rcParams.update({'font.size': 13})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
levels = np.arange(-0.5, 0.5, .01)
levels1 = np.arange(-200, 200, 10)

k1=-0.5
k2=0.5
axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


plt.figtext(0.125, 0.845, 'a)')
plt.figtext(0.55, 0.845, 'b)')
plt.figtext(0.125, 0.425, 'c)')
plt.figtext(0.55, 0.425, 'd)')

obs=(sst_main).sel(time=slice('1970-01-01','2010-12-30'))

nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)


# Things we are going to regress onto
x = np.arange(0,len(obs['time']),1)
y = obs['ts'].values

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err


obs['trends'] = (('lat', 'lon'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)



weights=np.cos(obs.lat* np.pi / 180.)*test
mylist_to=(obs['trends']*weights*test).sel(lat=slice(-60,60)).sum('lat',skipna=True) / (weights.sel(lat=slice(-60,60)).sum('lat',skipna=True))
mylist_to=mylist_to.mean('lon')
trends_nw=obs['trends']-mylist_to
uniform = np.full((ny, nx),1)*mylist_to.values
obs['uniform']=(('lat', 'lon'), uniform)

obs['trends_nw']=(('lat','lon'),trends_nw)


levels = np.arange(-0.5, 0.5, .01)
levels1 = np.arange(-200, 200, 10)

k1=-0.5
k2=0.5
axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[0].contourf(sst_main.lon, sst_main.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[0].set_title('simulated SST trend', fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[0], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
cb1.set_label('$^o$C/dec.', fontsize=13)

axlist[0].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
axlist[0].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[0].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([20, 100, 180, -100, -20])
gl.ylocator = mticker.FixedLocator([-50,-25,0,25,50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.ylabel_style = {'size': 15, 'color': 'gray'}

cf1=axlist[2].contourf(sst_main.lon, sst_main.lat, obs.uniform,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[2].set_title('uniform warming (T$_0$)', fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[2], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
cb1.set_label('$^o$C/dec.', fontsize=13)

axlist[2].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
axlist[2].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[2].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([20, 100, 180, -100, -20])
gl.ylocator = mticker.FixedLocator([-50,-25,0,25,50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.ylabel_style = {'size': 15, 'color': 'gray'}

#%%

from scipy import signal
from scipy import misc

corr=xr.corr(obs.trends_nw,sst_main.trends*(-1))

trends_array=(obs.trends_nw*test).values.flatten()
trends_array = trends_array[~np.isnan(trends_array)]

trends_pdo=(sst_main.trends*(-1)*test).values.flatten()
trends_pdo = trends_pdo[~np.isnan(trends_pdo)]



from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(trends_pdo,trends_array)


#fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(38, 25),subplot_kw={'projection': ccrs.PlateCarree(180)})

plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
levels = np.arange(-0.5, 0.5, .01)
levels1 = np.arange(-200, 200, 10)

k1=-0.5
k2=0.5
axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[3].contourf(sst_main.lon, sst_main.lat, obs.trends_nw-sst_main.trends*(-1)*slope,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[3].set_title('residual warming pattern', fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[3], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
cb1.set_label('$^o$C/dec.', fontsize=13)

axlist[3].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
axlist[3].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[3].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[3].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([20, 100, 180, -100, -20])
gl.ylocator = mticker.FixedLocator([-50,-25,0,25,50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.ylabel_style = {'size': 15, 'color': 'gray'}

cf1=axlist[1].contourf(sst_main.lon, sst_main.lat, sst_main.trends*(-1)*slope,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[1].set_title('a*PDO historical', fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[1], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
cb1.set_label('$^o$C/dec.', fontsize=13)

axlist[1].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
axlist[1].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[1].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([20, 100, 180, -100, -20])
gl.ylocator = mticker.FixedLocator([-50,-25,0,25,50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.ylabel_style = {'size': 15, 'color': 'gray'}

