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

    uas_4xCO2=forcing['psl']
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
    uas_regrid = uas_regrid.assign_coords(year=list(range(1850,a+1850)))
    
    return uas_regrid

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=output

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_BCC-CSM2-MR_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')
forcing = xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_BCC-CSM2-MR_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc')
forcing = xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_CAMS-CSM1-0_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

### load and concatenate data ###
#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_ACCESS-CM2_abrupt-4xCO2_r1i1p1f1_gn_095001-144912.nc')
forcing = xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)

output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

forcing = xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CAS-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-CM6-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CESM2_historical_r11i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CESM2-FV2_historical_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CESM2-WACCM_historical_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CESM2-WACCM-FV2_historical_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CIESM_historical_r1i1p1f1_gr_185001-201412.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-ESM2-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-CM6-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_CNRM-CM6-1-HR_historical_r1i1p1f2_gr_185001-201412.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

#control = xr.open_dataset('/Volumes/Extreme SSD/CMIP_data/tts_Amon_CNRM-ESM2-1_abrupt-4xCO2_r1i1p1f2_gr_185001-234912.nc')
forcing=xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output



forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_E3SM-1-0_historical_r1i1p1f1_gr_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_FGOALS-f3-L_historical_r1i1p1f1_gr_185001-201412.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_FGOALS-g3_historical_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_GFDL-CM4_historical_r1i1p1f1_gr1_*.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_*.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_GISS-E2-1-G_historical_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_GISS-E2-1-H_historical_r1i1p1f1_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_HadGEM3-GC31-MM_historical_r1i1p1f3_gn_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_INM-CM4-8_historical_r1i1p1f1_gr1_*.nc')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

# forcing=xr.open_mfdataset('/Volumes/Extreme SSD/CMIP_data/ts_Amon_INM-CM4-8_abrupt-4xCO2_r1i1p1f1_gr1_*.nc')
# a=int(forcing.sizes['time']/12)
# output=regrid_anomaly(forcing,a)
# mylist=xr.concat([mylist,output], 'new_dim')
# del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_INM-CM5-0_historical_r1i1p1f1_gr1_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_KACE-1-0-G_historical_r1i1p1f1_gr_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_MCM-UA-1-0_historical_r1i1p1f1_gn_185001-201412.nc')
forcing = forcing.rename({'longitude': 'lon', 'latitude': 'lat'})
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_MIROC-ES2L_historical_r1i1p1f2_gn_185001-201412.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_MPI-ESM-1-2-HAM_historical_r1i1p1f1_gn_*.nc')
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

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output



forcing1=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing1=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_NESM3_historical_r1i1p1f1_gn_185001-201412.nc')

forcing=xr.concat([forcing1],'time')
a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


forcing=xr.open_dataset('/Users/ullaklintheede/Downloads/psl_Amon_NorCPM1_historical_r23i1p1f1_gn_185001-201412.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_SAM0-UNICON_historical_r1i1p1f1_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_TaiESM1_historical_r1i1p1f1_gn_185001-201412.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output

forcing=xr.open_mfdataset('/Users/ullaklintheede/Downloads/psl_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_*.nc')

a=int(forcing.sizes['time']/12)
output=regrid_anomaly(forcing,a)
mylist=xr.concat([mylist,output], 'new_dim')
del output


mylist.to_netcdf('/Users/ullaklintheede/psl_historical_mean.nc')


