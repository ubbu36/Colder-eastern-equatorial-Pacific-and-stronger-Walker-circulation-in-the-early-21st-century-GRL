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

sst_main=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
sst=sst_main.rolling(time=5*12, center=True).mean().sel(time=slice('1920-01-01','2016-12-30'))

sst_low=sst['sst'].sel(lat=slice(65,20),lon=slice(120,260))


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
y = sst['sst'].values

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




obs=sst_main.sel(time=slice('1980-01-01','2021-12-30'))

nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)


# Things we are going to regress onto
x = np.arange(0,len(obs['time']),1)
y = obs['sst'].values

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err


obs['trends'] = (('lat', 'lon'), trends*10*12)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)



weights=np.cos(obs.lat* np.pi / 180.)
mylist_to=(obs['trends']*weights).sel(lat=slice(60,-60)).sum('lat',skipna=True) / (weights.sel(lat=slice(60,-60)).sum('lat',skipna=True))
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

axlist[0].set_title('observed SST trend', fontsize=16)
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


trends_array=obs.trends_nw.values.flatten()
trends_array = trends_array[~np.isnan(trends_array)]

trends_pdo=(sst_main.trends*(-1)).values.flatten()
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


corr1=xr.corr(obs.trends_nw.sel(lat=slice(60,-60)),sst_main.trends.sel(lat=slice(60,-60))*(-1))
corr2=xr.corr(obs.trends.sel(lat=slice(60,-60)),obs['uniform'].sel(lat=slice(60,-60)))
corr3=xr.corr(obs.trends_nw.sel(lat=slice(60,-60)),obs.trends_nw-sst_main.trends.sel(lat=slice(60,-60))*(-1)*slope)


plt.figtext(0.57, 0.8, 'corr= '+str("%.2f" %corr1.values))
#plt.figtext(0.15, 0.385,'corr= '+str("%.2f" %corr2.values))
plt.figtext(0.57, 0.385, 'corr= '+str("%.2f" %corr3.values))

fig.savefig('Heede_Fig3.pdf',dpi=400, bbox_inches = "tight")
# #%%
# filelist = glob.glob(os.path.join('/Volumes/Armor_CMIP6/', 'ts_historical*.nc'))
# filelist=sorted(filelist, key=lambda s: s.lower())
# mylist=xr.open_dataset(filelist[8],decode_cf=False)

# #mylist_control=xr.Dataset.to_array(mylist_control)*test.isel(lev=0)
# #mylist_control_OT=mylist_control.sel(variable='ts',new_dim=[0,13,14,15,32,33,36])
# mylist_control_OT=mylist['ts'].isel(ens_member=0)
# mylist_control_OT=mylist_control_OT.rolling(year=5, center=True).mean()

# sst_low=mylist_control_OT.sel(lat=slice(20,60),lon=slice(120,260))

# sst_low=sst_low.dropna("year")
# #st_low=sst_low.drop_vars("time")


# sst_low=sst_low.rename({'year': 'time'})
# sst_low=sst_low.transpose("time", "lon", "lat")

# solver = Eof(sst_low)

# eofs = solver.eofs(neofs=2)
# eofs=eofs.transpose("lat", "lon",'mode')
# pcs = solver.pcs(npcs=2)
# eofs_dec=eofs.sel(mode=0)*10

# nx=len(mylist_control_OT.lon)
# ny=len(mylist_control_OT.lat)

# # Data containers
# trends = np.full((ny, nx), np.NaN)
# p_values = np.full((ny, nx), np.NaN)
# trends_std_err = np.full((ny, nx), np.NaN)

# # Things we are going to regress onto
# x = pcs.sel(mode=1).values
# y = mylist_control_OT.squeeze().dropna("year").values

# # Good old for loop
# for j in range(ny):
#     for i in range(nx):
#         slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
#         trends[j, i] = slope
#         p_values[j, i] = p_value
#         trends_std_err[j, i] = std_err

# # Back to xarray


# mylist_control_OT['trends'] = (('lat', 'lon'), trends*13)
# mylist_control_OT['trends_std_err'] = (('lat', 'lon'), trends_std_err)
# mylist_control_OT['p_values'] = (('lat', 'lon'), p_values)

# levels = np.arange(-0.5, 0.5, .01)
# levels1 = np.arange(-200, 200, 10)

# k1=-0.5
# k2=0.5

# #plt.figtext(0.125, 0.34, 'e)')
# #plt.figtext(0.55, 0.31, 'f)')
# #fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(38, 25),subplot_kw={'projection': ccrs.PlateCarree(180)})
# #axlist = axarr.flatten()



# pdo=mylist_control_OT.trends

# #%%
# filelist = glob.glob(os.path.join('/Volumes/Armor_CMIP6/', 'ts_historical*.nc'))
# filelist=sorted(filelist, key=lambda s: s.lower())
# mylist=xr.open_dataset(filelist[8],decode_cf=False)
# mylist=mylist.rolling(year=1, center=True).mean().sel(year=slice(1970,2010)).dropna('year')
# #mylist_control=xr.Dataset.to_array(mylist_control)*test.isel(lev=0)
# #mylist_control_OT=mylist_control.sel(variable='ts',new_dim=[0,13,14,15,32,33,36])
# mylist_control_OT=mylist['ts'].isel(ens_member=0)
# #mylist_control_OT=mylist_control_OT.rolling(year=5, center=True).mean()

# nx=len(mylist_control_OT.lon)
# ny=len(mylist_control_OT.lat)

# # Data containers
# trends = np.full((ny, nx), np.NaN)
# p_values = np.full((ny, nx), np.NaN)
# trends_std_err = np.full((ny, nx), np.NaN)

# # Things we are going to regress onto
# x = mylist_control_OT['year'].values
# y = mylist_control_OT.squeeze().dropna("year").values

# # Good old for loop
# for j in range(ny):
#     for i in range(nx):
#         slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
#         trends[j, i] = slope
#         p_values[j, i] = p_value
#         trends_std_err[j, i] = std_err

# # Back to xarray


# mylist_control_OT['trends'] = (('lat', 'lon'), trends*10)
# mylist_control_OT['trends_std_err'] = (('lat', 'lon'), trends_std_err)
# mylist_control_OT['p_values'] = (('lat', 'lon'), p_values)

# obs=xr.open_dataset('/Users/ullaheede_1/Downloads/ersst.v4.1854-2020.nc')
# ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.0)),
#                      'lon': (['lon'], np.arange(0, 359, 1)),
#                     }
#                    )

# regridder = xe.Regridder(obs, ds_out, 'bilinear')
# mylist_obs_regrid = regridder(obs)
# mask_ocean = 1 * np.ones((mylist_obs_regrid.dims['lat'], mylist_obs_regrid.dims['lon'])) * np.isfinite(mylist_obs_regrid.sst.isel(time=0))
# test=mask_ocean.where(mask_ocean == 1)


# weights=np.cos(mylist_control_OT.lat* np.pi / 180.)
# mylist_to=(mylist_control_OT['trends']*weights*test.squeeze().values).sel(lat=slice(-60,60)).sum('lat',skipna=True) /(weights*test.squeeze()).sel(lat=slice(-60,60)).sum('lat',skipna=True)
# mylist_to=mylist_to.mean('lon')
# trends_nw=mylist_control_OT['trends']-mylist_to
# mylist_control_OT['trends_nw']=(('lat', 'lon'), trends_nw)

# uniform = np.full((ny, nx),1)*mylist_to.values
# mylist_control_OT['uniform']=(('lat', 'lon'), uniform)

# levels = np.arange(-0.5, 0.5, 0.01)
# levels1 = np.arange(-200, 200, 10)

# k1=-0.5
# k2=0.5

# corr=xr.corr(mylist_control_OT.trends_nw,pdo*(-1))


# trends_array=mylist_control_OT.trends_nw.values.flatten()
# trends_array = trends_array[~np.isnan(trends_array)]

# trends_pdo=(pdo*(-1)).values.flatten()
# trends_pdo = trends_pdo[~np.isnan(trends_pdo)]



# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(trends_pdo,trends_array)


# #plt.figtext(0.125, 0.34, 'e)')
# #plt.figtext(0.55, 0.31, 'f)')

# #fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(38, 25),subplot_kw={'projection': ccrs.PlateCarree(180)})
# #axlist = axarr.flatten()

# cf1=axlist[1].contourf(mylist_control_OT.lon, mylist_control_OT.lat, (mylist_control_OT.trends),levels, extend="both",
#              transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

# axlist[1].set_title('SST trend, 1970-2010 CESM2-FV2', fontsize=20)
# cb1 = fig.colorbar(cf1, ax=axlist[1], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
# cb1.set_label('SST ($^o$C)', fontsize=20)

# axlist[1].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
# axlist[1].coastlines()
# #plt.colorbar()
# #plt.clim(-0.7,0.7)
# axlist[1].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
# gl = axlist[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=2, color='gray', alpha=0, linestyle='--')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xlines = False
# gl.xlocator = mticker.FixedLocator([120, 180, -120, -60])
# gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 19, 'color': 'gray'}
# gl.ylabel_style = {'size': 19, 'color': 'gray'}

# cf1=axlist[3].contourf(mylist_control_OT.lon, mylist_control_OT.lat, pdo*slope*(-1),levels, extend="both",
#              transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

# axlist[3].set_title('a*PDO historical CESM2-FV2', fontsize=20)
# cb1 = fig.colorbar(cf1, ax=axlist[3], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
# cb1.set_label('SST ($^o$C)', fontsize=20)

# axlist[3].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
# axlist[3].coastlines()
# #plt.colorbar()
# #plt.clim(-0.7,0.7)
# axlist[3].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
# gl = axlist[3].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=2, color='gray', alpha=0, linestyle='--')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xlines = False
# gl.xlocator = mticker.FixedLocator([120, 180, -120, -60])
# gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 19, 'color': 'gray'}
# gl.ylabel_style = {'size': 19, 'color': 'gray'}
# #fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(38, 25),subplot_kw={'projection': ccrs.PlateCarree(180)})
# #axlist = axarr.flatten()

# cf1=axlist[7].contourf(mylist_control_OT.lon, mylist_control_OT.lat, (mylist_control_OT.trends_nw-pdo*slope*(-1)),levels, extend="both",
#              transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

# axlist[7].set_title('SST trend, 1970-2010 CESM2-FV2 minus T$_0$ and a*PDO', fontsize=20)
# cb1 = fig.colorbar(cf1, ax=axlist[7], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
# cb1.set_label('SST ($^o$C)', fontsize=20)

# axlist[7].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
# axlist[7].coastlines()
# #plt.colorbar()
# #plt.clim(-0.7,0.7)
# axlist[7].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
# gl = axlist[7].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=2, color='gray', alpha=0, linestyle='--')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xlines = False
# gl.xlocator = mticker.FixedLocator([120, 180, -120, -60])
# gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 19, 'color': 'gray'}
# gl.ylabel_style = {'size': 19, 'color': 'gray'}

# cf1=axlist[5].contourf(mylist_control_OT.lon, mylist_control_OT.lat, mylist_control_OT.uniform,levels, extend="both",
#              transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

# axlist[5].set_title('uniform warming (T$_0$), 1970-2010, CESM2-FV2', fontsize=20)
# cb1 = fig.colorbar(cf1, ax=axlist[5], orientation='vertical', shrink=0.7, pad=0.02,ticks=[-0.3, 0, 0.3])
# cb1.set_label('SST ($^o$C)', fontsize=20)

# axlist[5].set_extent([-1790, 790, -61, 61], ccrs.PlateCarree(180))
# axlist[5].coastlines()
# #plt.colorbar()
# #plt.clim(-0.7,0.7)
# axlist[5].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
# gl = axlist[5].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=2, color='gray', alpha=0, linestyle='--')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xlines = False
# gl.xlocator = mticker.FixedLocator([120, 180, -120, -60])
# gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 19, 'color': 'gray'}
# gl.ylabel_style = {'size': 19, 'color': 'gray'}
