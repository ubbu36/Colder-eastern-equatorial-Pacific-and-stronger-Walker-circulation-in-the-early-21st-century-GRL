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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pylab import *
plt.rcParams.update({'font.size': 32})
k=10

sst_low=xr.open_dataset('/Users/ullaheede_1/Downloads/sst.mnmean.nc')
#%%
obs1=sst_low.sel(time=slice('1980-01-01','2020-12-30'))
#%%
obs=obs1.assign_coords(time=range(0,len(obs1.time)))
#%%
nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['time'])
y = obs['sst'].values

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err
#%%
# Back to xarray


obs['trends'] = (('lat', 'lon'), trends*10*12)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)

weights=np.cos(obs.lat* np.pi / 180.)
mylist_to=(obs['trends']*weights).sel(lat=slice(60,-60)).sum('lat',skipna=True) / weights.sel(lat=slice(60,-60)).sum('lat',skipna=True)
mylist_to=mylist_to.mean('lon')
trends_nw=obs['trends']-mylist_to
obs['trends_nw']=trends_nw
#%%
plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
levels = np.arange(-0.31, 0.31, .01)
levels1 = np.arange(-200, 200, 10)

k1=-0.31
k2=0.31
fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(25, 20),subplot_kw={'projection': ccrs.PlateCarree(180)})


axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')

plt.figtext(0.125, 0.845, 'i)')
plt.figtext(0.55, 0.81, 'j)')
plt.figtext(0.125, 0.425, 'k)')
plt.figtext(0.55, 0.425, 'l)')

cf1=axlist[0].contourf(obs.lon, obs.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[0].set_title('SST trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[0], orientation='vertical', shrink=0.65, pad=0.02,ticks=[-0.3, 0, 0.3])
cb1.set_label('$^o$C/dec.', fontsize=20)

axlist[0].set_extent([-130, 120, -41, 41], ccrs.PlateCarree(180))
axlist[0].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[0].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}
axlist[0].add_patch(mpatches.Rectangle(xy=[80, -5], width=70, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )

axlist[0].add_patch(mpatches.Rectangle(xy=[200, -5], width=80, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )
#%%
sst_low=xr.open_dataset('/Users/ullaheede_1/Downloads/precip.mon.mean.nc')

obs1=sst_low.sel(time=slice('1980-01-01','2020-12-30'))

obs=obs1.groupby('time.year').mean('time')
obs=obs.fillna(0)
obs_l=obs['precip'].values


nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('lat', 'lon'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)


plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
levels = np.arange(-0.5, 0.5, .01)
levels1 = np.arange(-200, 200, 10)

k1=-0.5
k2=0.5


#fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[3].contourf(obs.lon, obs.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[3].set_title('Precipitation trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[3], orientation='vertical', shrink=0.65, pad=0.02,ticks=[-0.3, 0, 0.3])
cb1.set_label('mm/day/dec.', fontsize=20)

axlist[3].set_extent([-130, 120, -41, 41], ccrs.PlateCarree(180))
axlist[3].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[3].add_feature(cfeature.COASTLINE, zorder=100, edgecolor='k')
gl = axlist[3].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}
axlist[3].add_patch(mpatches.Rectangle(xy=[130, -5], width=30, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )

axlist[3].add_patch(mpatches.Rectangle(xy=[180, -5], width=100, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )


sst_low=xr.open_dataset('/Users/ullaheede_1/Downloads/olr.mon.mean.nc')

obs1=sst_low.sel(time=slice('1980-01-01','2020-12-30'))

obs=obs1.groupby('time.year').mean('time')
obs=obs.fillna(0)
obs_l=obs['olr'].values


nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('lat', 'lon'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)


plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
k1=-5
k2=5
levels = np.arange(k1, k2, 0.5)
levels1 = np.arange(-200, 200, 10)


#fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[2].contourf(obs.lon, obs.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[2].set_title('OLR trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[2], orientation='vertical', shrink=0.65, pad=0.02,ticks=[-4, 0, 4])
cb1.set_label('W/m$^2$/dec.', fontsize=20)

axlist[2].set_extent([-130, 120, -41, 41], ccrs.PlateCarree(180))
axlist[2].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[2].add_feature(cfeature.COASTLINE, zorder=100, edgecolor='k')
gl = axlist[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-40,-20,0,20,40])   
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

axlist[2].add_patch(mpatches.Rectangle(xy=[160, -10], width=80, height=20,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )



sst_low=xr.open_dataset('/Users/ullaheede_1/Downloads/slp.mon.mean.nc')

obs1=sst_low.sel(time=slice('1980-01-01','2020-12-30'))

obs=obs1.groupby('time.year').mean('time')
obs=obs.fillna(0)
obs_l=obs['slp'].values


nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, j, i])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('lat', 'lon'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)



plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
k1=-0.7
k2=0.7
levels = np.arange(k1, k2, 0.05)
levels1 = np.arange(-200, 200, 10)


#fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[1].contourf(obs.lon, obs.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[1].set_title('SLP trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[1], orientation='vertical', shrink=0.6, pad=0.02,ticks=[-0.5, 0, 0.5])
cb1.set_label('hPa/dec.', fontsize=20)

axlist[1].set_extent([-130, 120, -31, 31], ccrs.PlateCarree(180))
axlist[1].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[1].add_feature(cfeature.COASTLINE, zorder=100, edgecolor='k')
gl = axlist[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-30,-15,0,15,30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

axlist[1].add_patch(mpatches.Rectangle(xy=[130, -5], width=20, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )

axlist[1].add_patch(mpatches.Rectangle(xy=[180, -5], width=60, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )


#%%
ssh_background=xr.open_dataset('/Users/ullaheede_1/Downloads/zos_AVISO_L4_199210-201012 _1.nc')
ssh_background=ssh_background.mean('time')
ssh_background=ssh_background.rename_vars({'zos':'SLA'})

ssh_low=xr.open_dataset('/Users/ullaheede_1/Downloads/ssh_monthlyMean.nc')
obs1=ssh_low.sel(Time=slice('1980-01-01','2020-12-30'))

obs=obs1.groupby('Time.year').mean('Time',skipna=True)
obs=obs.transpose('year','Longitude','Latitude','bnds')

#obs=obs.fillna(0)
obs_l=obs['SLA'].values


nx=len(obs.Longitude)
ny=len(obs.Latitude)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, i, j])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('Latitude', 'Longitude'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)


#%%
plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
k1=0.005
k2=0.065
levels = np.arange(k1, k2, 0.002)
levels1 = np.arange(-200, 200, 10)


#fig, axarr = plt.subplots(nrows=2, ncols=5, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[4].contourf(obs.Longitude, obs.Latitude, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[4].set_title('SLH trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[4], orientation='vertical', shrink=0.6, pad=0.02,ticks=[0.01, 0.04, 0.06])
cb1.set_label('m/dec.', fontsize=20)

axlist[4].set_extent([-130, 120, -41, 41], ccrs.PlateCarree(180))
axlist[4].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[4].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[4].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-30,-15,0,15,30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

axlist[4].add_patch(mpatches.Rectangle(xy=[130, -5], width=50, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )

axlist[4].add_patch(mpatches.Rectangle(xy=[200, -5], width=80, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )


#%%

olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/uwnd.mon.mean.nc')
#olr_west=olr_low['uwnd'].isel(level=0).sel(lon=slice(150,280),lat=slice(5,-5)).mean('lat',skipna=True).mean('lon',skipna=True)

obs1=olr_low.sel(time=slice('1980-01-01','2020-12-30')).isel(level=0)

obs=obs1.groupby('time.year').mean('time',skipna=True)
obs=obs.transpose('year','lon','lat')

obs=obs.fillna(0)
obs_l=obs['uwnd'].values


nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, i, j])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('lat', 'lon'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)



plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
k1=-1
k2=1
levels = np.arange(k1, k2, 0.05)
levels1 = np.arange(-200, 200, 10)


#fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[5].contourf(obs.lon, obs.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[5].set_title('zonal wind trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[5], orientation='vertical', shrink=0.6, pad=0.02,ticks=[-0.4, 0, 0.4])
cb1.set_label('m/s/dec.', fontsize=20)

axlist[5].set_extent([-130, 120, -31, 31], ccrs.PlateCarree(180))
axlist[5].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[5].add_feature(cfeature.COASTLINE, zorder=100, edgecolor='k')
gl = axlist[5].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-30,-15,0,15,30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

axlist[5].add_patch(mpatches.Rectangle(xy=[150, -5], width=130, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )





#%%

oscar=xr.open_dataset('/Users/ullaheede_1/Downloads/oscar_monthlyMean.nc')
oscar=oscar.isel(time=slice(0,len(oscar.time)-30)).isel(depth=0)

obs1=oscar.sel(time=slice('1980-01-01','2020-12-30'))

obs=obs1.groupby('time.year').mean('time',skipna=True)
obs=obs.transpose('year','longitude','latitude')

obs=obs.fillna(0)
obs_l=obs['u'].values


nx=len(obs.longitude)
ny=len(obs.latitude)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, i, j])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('latitude', 'longitude'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)



plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
k1=-0.1
k2=0.1
levels = np.arange(k1, k2, 0.005)
levels1 = np.arange(-200, 200, 10)


#fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[6].contourf(obs.longitude, obs.latitude, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[6].set_title('zonal surface current trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[6], orientation='vertical', shrink=0.6, pad=0.02,ticks=[-0.075, 0, 0.075])
cb1.set_label('m/s/dec.', fontsize=20)

axlist[6].set_extent([-130, 120, -31, 31], ccrs.PlateCarree(180))
axlist[6].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[6].add_feature(cfeature.COASTLINE, zorder=100, edgecolor='k')
gl = axlist[6].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-30,-15,0,15,30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

axlist[6].add_patch(mpatches.Rectangle(xy=[150, -5], width=130, height=10,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )



#%%

olr_low=xr.open_dataset('/Users/ullaheede_1/Downloads/omega.mon.mean.nc')

obs1=olr_low.sel(time=slice('1980-01-01','2020-12-30')).sel(level=500)

obs=obs1.groupby('time.year').mean('time',skipna=True)
obs=obs.transpose('year','lon','lat')

obs=obs.fillna(0)
obs_l=obs['omega'].values


nx=len(obs.lon)
ny=len(obs.lat)

# Data containers
trends = np.full((ny, nx), np.NaN)
p_values = np.full((ny, nx), np.NaN)
trends_std_err = np.full((ny, nx), np.NaN)

# Things we are going to regress onto
x = np.asarray(obs['year'])
y = obs_l

# Good old for loop
for j in range(ny):
    for i in range(nx):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y[:, i, j])
        trends[j, i] = slope
        p_values[j, i] = p_value
        trends_std_err[j, i] = std_err

# Back to xarray
obs['trends'] = (('lat', 'lon'), trends*10)
obs['trends_std_err'] = (('lat', 'lon'), trends_std_err*10)
obs['p_values'] = (('lat', 'lon'), p_values)



plt.rcParams.update({'font.size': 15})
cmap = cmocean.cm.balance
#cmap='BrBG_r'
k1=-0.02
k2=0.02
levels = np.arange(k1, k2, 0.001)
levels1 = np.arange(-200, 200, 10)


#fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(25, 10),subplot_kw={'projection': ccrs.PlateCarree(180)})


#axlist = axarr.flatten()
#plt.figtext(0.125, 0.34, 'e)')
#plt.figtext(0.55, 0.31, 'f)')


cf1=axlist[7].contourf(obs.lon, obs.lat, obs.trends,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[7].set_title('omega 500 mb trend', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[7], orientation='vertical', shrink=0.6, pad=0.02,ticks=[-0.01, 0, 0.01])
cb1.set_label('hPa/s/dec.', fontsize=20)

axlist[7].set_extent([-130, 120, -41, 41], ccrs.PlateCarree(180))
axlist[7].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[7].add_feature(cfeature.COASTLINE, zorder=100, edgecolor='k')
gl = axlist[7].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-30,-15,0,15,30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

axlist[7].add_patch(mpatches.Rectangle(xy=[130, -10], width=20, height=20,
                                    edgecolor='black',
                                    facecolor='none',
                                    linewidth=4,
                                    alpha=1,
                                    transform=ccrs.PlateCarree())
                 )
