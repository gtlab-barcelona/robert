#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime, timedelta


# In[2]:


# load data from netCDF file
era5_file = Dataset('/sda/climate_anomalies/era5_global_tmp-pre_2000-21.nc')
print(era5_file)


# Accumulated liquid and frozen water, including rain and snow, that falls to the Earth's surface. It is the sum of large-scale precipitation (that precipitation which is generated by large-scale weather patterns, such as troughs and cold fronts) and convective precipitation (generated by convection which occurs when air at lower levels in the atmosphere is warmer and less dense than the air above, so it rises). Precipitation variables do not include fog, dew or the precipitation that evaporates in the atmosphere before it lands at the surface of the Earth. This variable is accumulated from the beginning of the forecast time to the end of the forecast step. The units of precipitation are depth in metres. It is the depth the water would have if it were spread evenly over the grid box. Care should be taken when comparing model variables with observations, because observations are often local to a particular point in space and time, rather than representing averages over a model grid box and model time step.

# In[18]:


# get the data
# NOTE: taking every 10th coord due to computing power limitations
pre = era5_file.variables['tp'][:]*100
lon = era5_file.variables['longitude'][:]
lat = era5_file.variables['latitude'][:]
print(pre.shape)
print(lon.shape) # from -180 to 179
print(lat.shape) # from -90 to 90 (including 0°)


# In[19]:


# import time var & convert into datetime object
time = era5_file.variables['time'][:]

dtime = []
start = datetime(1900, 1, 1)
for t in time:
    delta = timedelta(hours=int(t))
    dtime.append(start + delta)


# In[20]:


# write function to get indices of time intervals as we'll need this a couple
# of times
def get_indices(start, end):
    iout = []
    for i, time in enumerate(dtime):
        if time >= start and time <= end:
            iout.append(i)
    return iout


# In[21]:


# get indices of years 2000/01-2020/01, i.e., reference period
start = datetime(2000, 1, 1)
end = datetime(2019, 12, 31)
iavg = get_indices(start, end)

# compute AVERAGE temperature per month per grid cell in reference period
pre_avg = []
iend = iavg[-1]    # index of last month in ref period
for istart in iavg[:12]:    
    # loop over first 12 indices, then, compute mean from first to last index
    # using every 12th value! To include last month of reference period use "iend+1".
    monthly_mean = np.mean(pre[istart:iend+1:12], axis=0)
    pre_avg.append(monthly_mean)    # returns list of arrays of shape (12, 181, 360)
np.shape(pre_avg)


# In[22]:


# compute temperature ANOMALY with index of month for 2020/12 to 2021/08
start = datetime(2020, 12, 1)
end = datetime(2021, 8, 1)
ianom = get_indices(start, end)

pre_anom = []
for i in ianom:
    imonth = dtime[i].month -1  # get index of month (e.g., Jan = 0, Dec = 11)
    monthly_anomaly = pre[i] - pre_avg[imonth]  # subtract respective month from tmp average
    pre_anom.append(monthly_anomaly)
np.shape(pre_anom)


# In[31]:


# plotting anomalies

# The data is defined in lat/lon coordinate system, so PlateCarree() is the
# appropriate transformation choice.
# (https://scitools.org.uk/cartopy/docs/latest/tutorials/understanding_transform.html)
data_crs = ccrs.PlateCarree()
v = 0.7

llon, llat = np.meshgrid(lon, lat)
for i, idate in enumerate(ianom):
    print(f'plotting anomaly map {i+1}/{len(ianom)}...')
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-25, 70, 5, 70])
    ax.set_title(dtime[idate].strftime('%b %Y'))
    ax.add_feature(cfeature.COASTLINE)
    pcm = ax.pcolormesh(lon, lat, pre_anom[i], # alternatively: ax.contourf()
                        transform=data_crs, cmap='PuOr', vmin=-v, vmax=v)
    cb = fig.colorbar(pcm, orientation='horizontal', shrink=0.5)
    cb.set_label('precipitation anomaly [cm]')
    ax.text(0.55,-.05,'reference period: 2000/01 - 2019/12',
            {'fontsize': 5, 'alpha': 0.7},
            transform=plt.gca().transAxes) # in axis coordinates
    fig.savefig(
        f'../figs/dragonflies-catalonia/pre_{dtime[idate].year}-{dtime[idate].month}.png',
        dpi=600, bbox_inches='tight', facecolor='white',
    )


# In[33]:


# plotting absolute temperature
data_crs = ccrs.PlateCarree()

llon, llat = np.meshgrid(lon, lat)
count = 1
for i in ianom:
    print(f'plotting pre map {count}/{len(ianom)}...')
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-25, 70, 5, 70])
    ax.set_title(dtime[i].strftime('%b %Y'))
    ax.add_feature(cfeature.COASTLINE)
    pcm = ax.pcolormesh(lon, lat, pre[i], # alternatively: ax.contourf()
                        transform=data_crs, cmap='Blues', vmin=0, vmax=1)
    cb = fig.colorbar(pcm, orientation='horizontal', shrink=0.5)
    cb.set_label('precipitation [cm]')
    fig.savefig(
        f'../figs/dragonflies-catalonia/pre_{dtime[i].year}-{dtime[i].month}_abs.png',
        dpi=600, bbox_inches='tight', facecolor='white',
    )
    count += 1


# In[ ]:




