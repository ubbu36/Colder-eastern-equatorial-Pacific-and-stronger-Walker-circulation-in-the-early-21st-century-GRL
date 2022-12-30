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
             'HadGEM3-GC31-LL','HadGEM3-GC3-MM','INM-CM4-8','INM-CM5-0','IPSL-CM6A','KACE-1-0-G','MCM-UA-1-0','MIROC-ES2L','MIROC6','MPI-ESM-1-2-HAM','MPI-ESM1-2-LR','MPI-ESM1-2-HR',\
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

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-109912.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=output

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_ACCESS-ESM1-5_abrupt-4xCO2_r1i1p1f1_gn_010101-025012.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_BCC-CSM2-MR_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')
forcing = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_BCC-CSM2-MR_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_BCC-ESM1_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_BCC-CSM2-MR_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')
forcing = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CAMS-CSM1-0_abrupt-4xCO2_r1i1p1f1_gn_303001-317912.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

### load and concatenate data ###
#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CanESM5_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

forcing = xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CAS-ESM2-0_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-CM6-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_mfdataset('//Volumes/Extreme SSD/CMIP_data/ts_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('//Volumes/Extreme SSD/CMIP_data/ts_Amon_CESM2-FV2_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CESM2-WACCM_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CESM2-WACCM-FV2_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CIESM_abrupt-4xCO2_r1i1p1f1_gr_000101-015012.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-ESM2-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CMCC-CM2-SR5_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-CM6-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CNRM-CM6-1_abrupt-4xCO2_r1i1p1f2_gr_185001-199912.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CNRM-CM6-1-HR_abrupt-4xCO2_r1i1p1f2_gr_185001-199912.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-ESM2-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_CNRM-ESM2-1_abrupt-4xCO2_r1i1p1f2_gr_185001-199912.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output



forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_E3SM-1-0_abrupt-4xCO2_r1i1p1f1_gr_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_FGOALS-f3-L_abrupt-4xCO2_r2i1p1f1_gr_185001-200912.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_FGOALS-g3_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_GFDL-CM4_abrupt-4xCO2_r1i1p1f1_gr1_*.nc')

forcing=xr.concat([forcing1],'time')
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

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_GISS-E2-1-H_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_HadGEM3-GC31-LL_abrupt-4xCO2_r1i1p1f3_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_HadGEM3-GC31-MM_abrupt-4xCO2_r1i1p1f3_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_INM-CM4-8_abrupt-4xCO2_r1i1p1f1_gr1_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

# forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_INM-CM4-8_abrupt-4xCO2_r1i1p1f1_gr1_*.nc')
# a=int(forcing.sizes['time']/12)
# output=regrid_anomaly(forcing,a)
# mylist=xr.concat([mylist,output], 'new_dim')
# del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_INM-CM5-0_abrupt-4xCO2_r1i1p1f1_gr1_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_IPSL-CM6A-LR_abrupt-4xCO2_r1i1p1f1_gr_185001-214912.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_KACE-1-0-G_abrupt-4xCO2_r1i1p1f1_gr_185001-200012.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MCM-UA-1-0_abrupt-4xCO2_r1i1p1f1_gn_*.nc')
forcing = forcing.rename({'longitude': 'lon', 'latitude': 'lat'})
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MIROC-ES2L_abrupt-4xCO2_r1i1p1f2_gn_185001-199912.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MIROC6_abrupt-4xCO2_r1i1p1f1_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_185001-186912.nc')
forcing2=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_187001-188912.nc')
forcing3=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_189001-190912.nc')
forcing4=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_191001-192912.nc')
forcing5=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_193001-194912.nc')
forcing6=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_195001-196912.nc')
forcing7=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_197001-198912.nc')
forcing8=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM-1-2-HAM_abrupt-4xCO2_r1i1p1f1_gn_199001-199912.nc')

forcing=xr.concat([forcing1,forcing2, forcing3,forcing4,forcing5,forcing6, forcing7, forcing8],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

#forcing1=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-HR_abrupt-4xCO2_r1i1p1f1_gn_*.nc')

#forcing=xr.concat([forcing1],'time')
#a=int(forcing.sizes['time']/12)
#output=regrid_anomaly(forcing,a)
#mylist=xr.concat([mylist,output], 'new_dim')
#del output

forcing1=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_185001-186912.nc')
forcing2=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_187001-188912.nc')
forcing3=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_189001-190912.nc')
forcing4=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_191001-192912.nc')
forcing5=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_193001-194912.nc')
forcing6=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_195001-196912.nc')
forcing7=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_197001-198912.nc')
forcing8=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_199001-200912.nc')
forcing9=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_201001-201412.nc')

forcing=xr.concat([forcing1,forcing2, forcing3,forcing4,forcing5,forcing6, forcing7, forcing8,forcing9],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output



forcing1=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_MRI-ESM2-0_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_NESM3_abrupt-4xCO2_r1i1p1f1_gn_185001-199912.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


forcing1=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_NorCPM1_abrupt-4xCO2_r1i1p1f1_gn_000101-008012.nc')
forcing2=xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_NorCPM1_abrupt-4xCO2_r1i1p1f1_gn_008101-015012.nc')
forcing=xr.concat([forcing1,forcing2],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_SAM0-UNICON_abrupt-4xCO2_r1i1p1f1_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_TaiESM1_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_UKESM1-0-LL_abrupt-4xCO2_r1i1p1f2_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

mylist_mean=mylist

mylist_mean.to_netcdf('/Users/ullaklintheede/ts_CMIP_abrupt-4xCO2_dim_mean.nc')

