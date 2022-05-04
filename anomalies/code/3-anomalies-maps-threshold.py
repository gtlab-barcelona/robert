# %% [markdown]
# # Anomaly maps
# 
# In this notebook, I create anomaly maps of multiple climatic variables to visually understand the population outbreaks in the Fringilla insect data.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime, timedelta

# %% [markdown]
# ### Importing CRU TS data
# 
# The Climate Research Unit gridded Time Series (CRU TS) is a widely used climate dataset with monthly resolution on a 0.5° latitude by 0.5° longitude grid over all land domains of the world except Antarctica. The dataset is introduced in this [paper](https://doi.org/10.1038/s41597-020-0453-3).

# %%
# load meta data from netCDF file
# temperature:
cru_ts_tmp = Dataset('/sda/climate-anomalies/cru_ts4.05.1901.2020.tmp.dat.nc')
print(cru_ts_tmp)

# %%
# precipitation:
cru_ts_pre = Dataset('/sda/climate-anomalies/cru_ts4.05.1901.2020.pre.dat.nc')
print(cru_ts_pre)

# %%
# import time var & convert into datetime object
time = cru_ts_tmp.variables['time'][:]

dtime = []
start = datetime(1900, 1, 1)
for t in time:
    delta = timedelta(days=int(t))
    dtime.append(start + delta)

# %%
# get the data
# NOTE: taking every s-th coord due to computing power limitations
s = 1
pre = cru_ts_pre.variables['pre'][:,::s,::s]
tmp = cru_ts_tmp.variables['tmp'][:,::s,::s]
lon = cru_ts_tmp.variables['lon'][::s]
lat = cru_ts_tmp.variables['lat'][::s]
print(tmp.shape)
print(pre.shape)
print(lon.shape)
print(lat.shape)

# %% [markdown]
# ### Computing anomalies

# %%
# define reference period and get respective indices in time list
ref_start = datetime(1991, 1, 16)
ref_end   = datetime(2020, 12, 16)

iref_start = dtime.index(ref_start)
iref_end = dtime.index(ref_end) + 1 # indexing in python does not include stop value 

print(iref_start, iref_end)

# %%
# compute monthly mean and sample standard deviation within reference period
def monthly_stats(var):
    '''
    var: climatic variable  
    '''
    month_mean = []
    month_std = []
    # compute stats individually for every month
    for imonth in range(12):
        month_mean.append(np.mean(var[iref_start+imonth:iref_end:12], axis=0))
        # maximum likelihood estimate of the variance for normally distributed variables
        month_std.append(np.std(var[iref_start+imonth:iref_end:12], axis=0))
    return month_mean, month_std

pre_mean, pre_std = monthly_stats(pre)
tmp_mean, tmp_std = monthly_stats(tmp)

# %%
# select time frame of interest (i. e., the anomaly maps are created for these years)
# NOTE: for this to work flawless, do not change month and day values!
first_year = 1998
final_year = 2020

anom_start = datetime(first_year, 1, 16)
anom_end = datetime(final_year, 12, 16)

ianom_start = dtime.index(anom_start)
ianom_end = dtime.index(anom_end) + 1 # indexing in python does not include stop value 

print(ianom_start, ianom_end)

# %%
# compute z-score (observation - mean / sample std) for time frame of interest
def zscore(var, mean, std):
    zscr = []
    # get int of first month in time frame of interest
    imonth = dtime[ianom_start].month - 1
    for obsv in var[ianom_start:ianom_end]:
        # to cycle through the months, compute remainder of division by 12
        imonth_loop = (imonth % 12)
        # compute z-score
        zscr.append((obsv - mean[imonth_loop]) / std[imonth_loop])
        imonth += 1
    return zscr

pre_zscr = zscore(pre, pre_mean, pre_std)
tmp_zscr = zscore(tmp, tmp_mean, tmp_std)
pre_zscr

# %% [markdown]
# ### Plotting

# %%
import numpy.ma as ma

# remove all z-scores which absolute value is smaller (or equal) than threshold,
# threshold is in units of standard deviation, remember the "68-95-99.7 rule"
threshold = 1

# (-threshold <= vals <= threshold) are masked!
def filter_small_values(var):
    var_rm = []
    for i in range(len(var)):
        # mask values inside interval
        var_rm.append(ma.masked_inside(var[i], threshold*-1, threshold))
    return var_rm

pre_zscr_rm = filter_small_values(pre_zscr)
tmp_zscr_rm = filter_small_values(tmp_zscr)

# %%
# The data is defined in lat/lon coordinate system, so PlateCarree() is the
# appropriate transformation choice.
# (https://scitools.org.uk/cartopy/docs/latest/tutorials/understanding_transform.html)
data_crs = ccrs.PlateCarree()

def plot_anom_maps(var, kind, cmap):
    number_of_plots = (final_year + 1) - first_year
    idummy = 0
    # collection of subplots for every year of time frame of interest
    for iplot in range(number_of_plots):
        current_year = dtime[ianom_start+idummy].strftime('%Y')
        print(f"{kind}: plotting #{iplot+1} of {number_of_plots}...")
        # as there are lots of figures and we save them directly anyways, let's
        # clear them from memory:
        # https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots#55834853
        fig, axs = plt.subplots(
            nrows=3, ncols=4, figsize=(16,9),
            subplot_kw={'projection':ccrs.Miller()}, # determine map projection
            num=1, clear=True # use same figure, but cleared (prevents memory overflow)
        )
        fig.suptitle(f"{current_year}: {kind} anomaly",
                     fontsize=20)
        # plotting individual month
        for ax in axs.flat:
            ax.set_extent([-25, 70, 5, 70])
            ax.set_title(dtime[ianom_start+idummy].strftime('%b'))
            pcm = ax.pcolormesh(
                lon, lat, var[idummy],
                transform=data_crs, cmap=cmap,
                rasterized=True, shading='nearest', # these settings are imprtant
                                                    # for properly rendering pdf
                vmin=-3, vmax=3
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            idummy += 1
        cbar = plt.colorbar(pcm, ax=axs[:,:], shrink=0.9)
        cbar.set_label('z-score [std] | values in '
                       + rf'[$-{threshold}\sigma$,$+{threshold}\sigma$]' 
                       + ' removed', fontsize=15)
        plt.figtext(x=0.76, y=0.11, 
                    s=f'reference period:\n{ref_start.year}-{ref_end.year}',
                    fontdict={'alpha': 0.7})
        fig.savefig(
            f'../figs/kaliningrad/maps/{kind}-{current_year}.pdf', bbox_inches='tight'
        )
        # due to high memory usage nonetheless, added the following, just in case:
        fig.clear()
        plt.close(fig)

# plotting...
plot_anom_maps(tmp_zscr_rm, "temperature", "coolwarm")
plot_anom_maps(pre_zscr_rm, "precipitation", "PuOr")