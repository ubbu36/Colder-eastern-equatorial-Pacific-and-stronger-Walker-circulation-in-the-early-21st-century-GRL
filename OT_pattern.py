#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:22:35 2022

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
mylist_to=(obs['trends']*weights).sel(lat=slice(65,-65)).sum('lat',skipna=True) / (weights.sel(lat=slice(65,-65)).sum('lat',skipna=True))
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



#%%

from scipy import signal
from scipy import misc

corr=xr.corr(obs.trends_nw,sst_main.trends*(-1))

trends_array=obs.trends_nw.values.flatten()
trends_array = trends_array[~np.isnan(trends_array)]

trends_pdo=(sst_main.trends*(-1)).values.flatten()
trends_pdo = trends_pdo[~np.isnan(trends_pdo)]


from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(trends_pdo,trends_array)


#%%

pdo=obs.trends_nw-sst_main.trends*(-1)*slope

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-88, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )
regridder = xe.Regridder(pdo, ds_out, 'bilinear')
trends_pdosub = regridder(pdo)

#%%

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
mask_ocean=mask_ocean

mylist = xr.open_dataset('/Users/ullaheede_1/Downloads/ts_OT_4x_dim_mean.nc')


mylist_control = xr.open_dataset('/Users/ullaheede_1/Downloads/ts_OT_control_dim_mean.nc')


lon=mylist.lon
lat=mylist.lat

weights=np.cos(lat* np.pi / 180.)


mylist = mylist-mylist_control
mylist = mylist*test
weights=np.cos(lat* np.pi / 180.)*test
mylist_to=(mylist*weights).sel(lat=slice(-65,65)).sum('lat',skipna=True) / weights.sel(lat=slice(-65,65)).sum('lat',skipna=True)
#mylist_to=(mylist*weights).sel(lat=slice(-90,90)).sum('lat',skipna=True) / weights.sel(lat=slice(-90,90)).sum('lat',skipna=True)

#mylist_to=mylist_to.sel(lon=slice(80,300))
mylist_ga=mylist_to.mean('lon',skipna=True)

mylist=mylist-mylist_ga
#mylist=xr.Dataset.to_array(mylist)

plot0 = mylist['ts'].isel(year=slice(0,9)).mean('year').mean('new_dim')

mylist = xr.open_dataset('/Users/ullaheede_1/Downloads/ts_OT_1pct_dim_mean.nc')


mylist_control = xr.open_dataset('/Users/ullaheede_1/Downloads/ts_OT_control_dim_mean.nc')


lon=mylist.lon
lat=mylist.lat

weights=np.cos(lat* np.pi / 180.)


mylist = mylist-mylist_control
mylist = mylist*test
weights=np.cos(lat* np.pi / 180.)*test
mylist_to=(mylist*weights).sel(lat=slice(-65,65)).sum('lat',skipna=True) / weights.sel(lat=slice(-65,65)).sum('lat',skipna=True)
#mylist_to=(mylist*weights).sel(lat=slice(-90,90)).sum('lat',skipna=True) / weights.sel(lat=slice(-90,90)).sum('lat',skipna=True)

#mylist_to=mylist_to.sel(lon=slice(80,300))
mylist_ga=mylist_to.mean('lon',skipna=True)

mylist=mylist-mylist_ga
#mylist=xr.Dataset.to_array(mylist)

plot0_ramp = mylist['ts'].isel(year=slice(10,30)).mean('year').mean('new_dim')


lon=plot0.lon
lat=plot0.lat

cmap = cmocean.cm.balance


plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'hatch.color': 'grey'})   

k1=-1.6
k2=1.6

levels = np.arange(k1, k2, 0.1)
#%%
fig, axarr = plt.subplots(nrows=4, ncols=1, figsize=(13, 19),subplot_kw={'projection': ccrs.PlateCarree(180)})
plt.figtext(0.13, 0.885, 'a)')
plt.figtext(0.13, 0.69, 'b)')
plt.figtext(0.13, 0.495, 'c)')
plt.figtext(0.13, 0.29, 'd)')

axlist = axarr.flatten()

cf1=axlist[0].contourf(trends_pdosub.lon, trends_pdosub.lat, trends_pdosub*4,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)
#cs = axlist[0].contourf(lon, lat, mylist_cat1_pos.isel(lev=0),alpha=0.01, extend="both",
#             transform=ccrs.PlateCarree(),hatches=['.'])
#cs = axlist[0].contourf(lon, lat, mylist_cat1_neg.isel(lev=0),alpha=0.01, extend="both",
#             transform=ccrs.PlateCarree(),hatches=['.'])           
      
axlist[0].set_title('observed NH-IWP warming pattern', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[0], orientation='vertical', shrink=0.8, pad=0.02)
cb1.set_label('$\Delta$SST ($^o$C)')

axlist[0].set_extent([-180, 179, -65, 65], ccrs.PlateCarree(180))
axlist[0].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[0].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -80])
gl.ylocator = mticker.FixedLocator([-60,-40,-20,0,20,40,60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}

cf1=axlist[1].contourf(lon, lat, plot0,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)
#cs = axlist[0].contourf(lon, lat, mylist_cat1_pos.isel(lev=0),alpha=0.01, extend="both",
#             transform=ccrs.PlateCarree(),hatches=['.'])
#cs = axlist[0].contourf(lon, lat, mylist_cat1_neg.isel(lev=0),alpha=0.01, extend="both",
#             transform=ccrs.PlateCarree(),hatches=['.'])           
      
axlist[1].set_title('OT pattern, abrupt4xCO2, years 1:10', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[1], orientation='vertical', shrink=0.8, pad=0.02)
cb1.set_label('$\Delta$SST ($^o$C)')

axlist[1].set_extent([-180, 179, -65, 65], ccrs.PlateCarree(180))
axlist[1].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[1].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -80])
gl.ylocator = mticker.FixedLocator([-60,-40,-20,0,20,40,60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}


cf1=axlist[2].contourf(lon, lat, plot0_ramp,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)
#cs = axlist[0].contourf(lon, lat, mylist_cat1_pos.isel(lev=0),alpha=0.01, extend="both",
#             transform=ccrs.PlateCarree(),hatches=['.'])
#cs = axlist[0].contourf(lon, lat, mylist_cat1_neg.isel(lev=0),alpha=0.01, extend="both",
#             transform=ccrs.PlateCarree(),hatches=['.'])           
      
axlist[2].set_title('OT pattern, 1pctCO2, years 10:30', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[2], orientation='vertical', shrink=0.8, pad=0.02)
cb1.set_label('$\Delta$SST ($^o$C)')

axlist[2].set_extent([-180, 179, -65, 65], ccrs.PlateCarree(180))
axlist[2].coastlines()
#plt.colorbar()
#plt.clim(-0.7,0.7)
axlist[2].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
gl = axlist[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlocator = mticker.FixedLocator([60,120, 180, -120, -60])
gl.ylocator = mticker.FixedLocator([-60,-40,-20,0,20,40,60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 19, 'color': 'gray'}
gl.ylabel_style = {'size': 19, 'color': 'gray'}


#%%

k=10

obss=xr.open_dataset('/Users/ullaheede_1/Downloads/ersst.v4.1854-2020.nc')
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )

regridder = xe.Regridder(obss, ds_out, 'bilinear')
mylist_obs_regrid = regridder(obss)
mask_ocean = 1 * np.ones((mylist_obs_regrid.dims['lat'], mylist_obs_regrid.dims['lon'])) * np.isfinite(mylist_obs_regrid.sst.isel(time=0))
test=mask_ocean.where(mask_ocean == 1)




mylist_OT=xr.open_dataset('/Users/ullaheede_1/Downloads/ts_OT_hist_dim_mean.nc',decode_cf=False)


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




cf1=axlist[3].contourf(plot3.lon,plot3.lat,plot3*3.5,levels, extend="both",
             transform=ccrs.PlateCarree(),vmin=k1,vmax=k2, cmap=cmap)

axlist[3].set_title('OT models mean historical trend minus T$_0$ ', fontsize=20)
cb1 = fig.colorbar(cf1, ax=axlist[3], orientation='vertical', shrink=0.8, pad=0.02)
#cb1.set_label('$^o$C/dec.', fontsize=13)

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
cb1.set_label('$\Delta$SST ($^o$C)')

trends_array=(trends_pdosub.sel(lat=slice(-60,60))).values.flatten()
trends_array = trends_array[~np.isnan(trends_array)]

trends_pdo=(plot0.sel(lat=slice(-60,60))).values.flatten()
trends_pdo = trends_pdo[~np.isnan(trends_pdo)]



from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(trends_pdo,trends_array)

#plt.plot(trends_array)
#plt.plot(trends_pdo)

corr1=xr.corr(plot0.sel(lat=slice(-60,60)),trends_pdosub.sel(lat=slice(-60,60)))
corr2=xr.corr(trends_pdosub.sel(lat=slice(-60,60)),plot0_ramp.sel(lat=slice(-60,60)))
corr3=xr.corr(trends_pdosub.sel(lat=slice(-60,60)),plot3.sel(lat=slice(-60,60)))


plt.figtext(0.20, 0.66, 'corr= '+str("%.2f" %corr1.values))
plt.figtext(0.20, 0.46,'corr= '+str("%.2f" %corr2.values))
plt.figtext(0.20, 0.26, 'corr= '+str("%.2f" %corr3.values))

fig.savefig('Heede_Fig4.pdf',dpi=400, bbox_inches = "tight")
