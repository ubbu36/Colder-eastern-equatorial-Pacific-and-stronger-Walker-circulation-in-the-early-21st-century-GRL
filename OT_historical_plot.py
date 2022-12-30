#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:17:52 2022

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

k=10

obss=xr.open_dataset('/Users/ullaklintheede/Downloads/ersst.v4.1854-2020.nc')
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )

regridder = xe.Regridder(obss, ds_out, 'bilinear')
mylist_obs_regrid = regridder(obss)
mask_ocean = 1 * np.ones((mylist_obs_regrid.dims['lat'], mylist_obs_regrid.dims['lon'])) * np.isfinite(mylist_obs_regrid.sst.isel(time=0))
test=mask_ocean.where(mask_ocean == 1)




mylist_OT=xr.open_dataset('/Users/ullaklintheede/ts_OT_hist_dim_mean.nc',decode_cf=False)


obs1=mylist_OT.sel(year=slice(1980,2015)).isel(new_dim=0)

#obs=obs1.groupby('time.year').mean('time')
obs2=obs1.fillna(0)
obs_l=(obs2['ts']).values

nx=len(obs2.lon)
ny=len(obs2.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs2['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:,j,i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err
        
obs2['trends'] = (('lat', 'lon'), trends*10)
obs2['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs2['p_values'] = (('lat', 'lon'), p_values)

weights=np.cos(obs2.lat* np.pi / 180.)
mylist_to=(test.squeeze()*obs2['trends']*weights).sel(lat=slice(-60,60)).sum('lat',skipna=True) / (weights*test.squeeze()).sel(lat=slice(-60,60)).sum('lat',skipna=True)
mylist_to=mylist_to.mean('lon')
trends_nw=obs2['trends']-mylist_to
obs2['trends_nw']=trends_nw
uniform = np.full((ny, nx),1)*mylist_to.values
obs2['uniform']=(('lat', 'lon'), uniform)

for x in range(1,8):
    obs1=mylist_OT.sel(year=slice(1980,2015)).isel(new_dim=x)

    #obs=obs1.groupby('time.year').mean('time')
    obs3=obs1.fillna(0)
    obs_l=(obs3['ts']).values

    nx=len(obs3.lon)
    ny=len(obs3.lat)

# Data containers
    trends = np.full((ny, nx), np.NaN)
    p_values = np.full((ny, nx), np.NaN)
    trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
    x = np.asarray(obs3['year'])
    y = obs_l

# Good old for loop
    for j in range(ny):
        for i in range(nx):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:,j,i])
            trends[j, i] = slope
            p_values[j, i] = p_value
            trends_std_err[j, i] = std_err
            
    obs3['trends'] = (('lat', 'lon'), trends*10)
    obs3['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
    obs3['p_values'] = (('lat', 'lon'), p_values)

    weights=np.cos(obs3.lat* np.pi / 180.)
    mylist_to=(test.squeeze()*obs3['trends']*weights).sel(lat=slice(-60,60)).sum('lat',skipna=True) / (weights*test.squeeze()).sel(lat=slice(-60,60)).sum('lat',skipna=True)
    mylist_to=mylist_to.mean('lon')
    trends_nw=obs3['trends']-mylist_to
    obs3['trends_nw']=trends_nw
    uniform = np.full((ny, nx),1)*mylist_to.values
    obs3['uniform']=(('lat', 'lon'), uniform)
    obs2=xr.concat([obs2,obs3],'new_dim')
    
plot1=obs2['trends'].mean('new_dim')
plot2=obs2['uniform'].mean('new_dim')
plot3=obs2['trends_nw'].mean('new_dim')




#%%
fig, axarr = plt.subplots(nrows=3, ncols=1, figsize=(8, 12),subplot_kw={'projection': ccrs.PlateCarree(180)})
axlist = axarr.flatten()
plt.rcParams.update({'font.size': 15})

levels = np.arange(-0.5, 0.5, .01)
k1=-0.5
k2=0.5
cmap = cmocean.cm.balance

plt.figtext(0.112, 0.85, 'a)')
plt.figtext(0.112, 0.58, 'b)')
plt.figtext(0.112, 0.32, 'c)')

cf1=axlist[0].contourf(plot1.lon,plot1.lat,plot1,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[0].set_title('OT models mean trend ', fontsize=20)
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
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([20, 100, 180, -100, -20])
gl.ylocator = mticker.FixedLocator([-50,-25,0,25,50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.ylabel_style = {'size': 15, 'color': 'gray'}

cf1=axlist[1].contourf(plot2.lon,plot2.lat,plot2,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[1].set_title('OT models uniform warming (T$_0$) ', fontsize=20)
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

#levels = np.arange(-1.5, 1.5, .01)
#k1=-1.5
#k2=1.5

cf1=axlist[2].contourf(plot3.lon,plot3.lat,plot3,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[2].set_title('OT models mean trend minus T$_0$ ', fontsize=20)
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
