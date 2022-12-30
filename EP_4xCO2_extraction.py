#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:29:28 2020

@author: ullaheede
"""

# module import
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import zarr as zarr

model_names=['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','BCC-ESM1','CAMS-CSM1-0','CanESM5','CAS-ESM2-0','CESM2','CESM2-FV','CESM2-WACCM','CESM2-WACCM-FV2',\
             'CIESM','CMCC-CM2-SR5','CNRM-CM6','CNRM-CM6-HR','CNRM-ESM2-1','E3SM','FGOALS-f3-L','FGOALS-g3','GFDL-CM4','GFDL-ESM4','GISS-E2-1-G','GISS-E2-1-H',\
             'HadGEM3-GC31-LL','HadGEM3-GC3-MM','INM-CM4-8','INM-CM5-0','IPSL-CM6A','KACE-1-0-G','MCM-UA-1-0','MIROC-ES2L','MIROC6','MPI-ESM-1-2-HAM','MPI-ESM1-2-LR',\
                 'MRI-ESM2','NESM3','NorCPM1','SAM0-UNICORN','TaiESM1','UKESM1-0-LL']


def regrid_anomaly(forcing,a):
#control 


 #   uas_control= control['U']

#4xCO2

    uas_4xCO2=forcing['ts']
   # uas_4xCO2=forcing['U']
 
    uas_4xCO2_anom=uas_4xCO2#-control_timemean


   #uas_4xCO2_anom_an=uas_4xCO2_anom
    uas_4xCO2_anom_an=uas_4xCO2_anom.groupby('time.year').mean('time')

    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.0)),
                     'lon': (['lon'], np.arange(0, 359, 1)),
                    }
                   )

    regridder = xe.Regridder(uas_4xCO2_anom_an, ds_out, 'bilinear')

    uas_regrid = regridder(uas_4xCO2_anom_an)
    uas_regrid = uas_regrid.assign_coords(year=list(range(a)))
    
    return uas_regrid


### load and concatenate data ###
#control = xr.open_dataset('/Users/ullaklintheede/Downloads/ts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CanESM5_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=output

#control = xr.open_dataset('/Users/ullaklintheede/Downloads/ts_Amon_CNRM-CM6-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_mfdataset('//Volumes/Extreme SSD/CMIP_data/ts_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CESM2-WACCM_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

#control = xr.open_dataset('/Users/ullaklintheede/Downloads/ts_Amon_CNRM-ESM2-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CMCC-CM2-SR5_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_E3SM-1-0_abrupt-4xCO2_r1i1p1f1_gr_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_FGOALS-g3_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_GFDL-ESM4_abrupt-4xCO2_r1i1p1f1_gr1_*.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_GISS-E2-1-G_abrupt-4xCO2_r1i1p3f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MIROC6_abrupt-4xCO2_r1i1p1f1_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_SAM0-UNICON_abrupt-4xCO2_r1i1p1f1_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

mylist_mean=mylist

mylist_mean.to_netcdf('/Users/ullaklintheede/ts_EP_abrupt-4xCO2_dim_mean.nc')

