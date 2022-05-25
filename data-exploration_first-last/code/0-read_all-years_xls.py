# %%
'''
read butterfly data and explore dataset

THIS CODE IS ADJUSTED FOR DEBUGGING V. CARDUI

TODO:
use seaborn for plotting
write function to extract data of all years and combine in single dataframe!
'''
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid", palette='Dark2')
#plt.style.use('bmh')
# %%
path = r'C:\Users\hotte\Desktop\Research\data\Lepidoptera_records_Courish_Spit_no_gender.xls'

# importing single year
df = pd.read_excel(
    io=path,
    sheet_name='Vanessa cardui',
    header=[0,1], # row to use for column labels
    skipfooter=2,
    #index_col=0, # make date the index of data frame
    dtype={('Year', 'Date'): str} # save date as string, otherwise (01.10 -> 1.1)
)

df = df.set_index(('Year', 'Date')).drop(labels='Total', axis=1)
df
# %%
# working with MultiIndex
# https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#

df_upd = df.stack(0)

# labelling index column holding single days and convert index to column
df_upd.index.set_names('Day', level=0, inplace=True)
df_upd.index.set_names('Year', level=1, inplace=True)
df_upd = df_upd.reset_index()

# converting to datetime object
df_upd['datetime'] = pd.to_datetime(
    arg = df_upd[['Year', 'Day']].astype(str).apply(' '.join, axis=1),
    format = '%Y %d.%m'
)
# and setting datetime as index and sorting
df_upd = df_upd.set_index(keys='datetime').sort_index(axis=0)

# save dataset
#df_upd.to_csv(r'../data/updated_butterfly_data.csv')
df_upd
# %%
# plotting
cols = ['L-1','L-2','L-3','L-4','L-5','L-7','L-8']
#cols = ['L-5']

def highlight_years(ax):
    for year in range(1982,2022,2):
        ax.axvspan(f'{year}-01-01', f'{year}-12-31', facecolor='black', edgecolor='none', alpha=0.1)

# single plot -> too many overlays
# df_upd[cols].plot()

# using subplots and pandas plotting
fig, axs = plt.subplots(len(cols), 1, figsize=(12,9), sharex=True)
df_upd[cols].plot(subplots=True, ax=axs)
axs[3].set_ylabel('counts')
#axs[len(cols)-1].set_xlabel('time')
#fig.suptitle('test')
for i in range(len(axs)):
    highlight_years(axs[i])
    axs[i].grid(visible=False, axis='x')
    #axs[i].set_yscale('log')
    axs[i].set_xlim(left='1982-01-01',right='2020-12-31') # OPTIONAL!
fig.tight_layout()
fig.savefig('../figs/time-evol_single-traps.pdf')
# use seaborn...
#%%
fig, axs = plt.subplots()
df_upd[cols].plot.area(stacked=False, ax=axs)
fig.tight_layout()
fig.savefig('../figs/time-evol_stacked-traps.pdf')
#%%
# colormap
import cmasher as cmr
colors = cmr.take_cmap_colors('Dark2', len(cols), return_fmt='hex')

fig, axs = plt.subplots(7, 1, figsize=(12,9), sharex=True)
for i in range(len(axs)):
    axs[i].scatter(df_upd.index, df_upd[cols[i]], label=cols[i], c=colors[i], marker='.')
    axs[i].legend()
    highlight_years(axs[i])
    axs[i].grid(visible=False, axis='x')
    #axs[i].set_yscale('log')
    axs[i].set_xlim(left=pd.to_datetime('1982-01-01'),right=pd.to_datetime('2020-12-31')) # OPTIONAL!
axs[3].set_ylabel('# of butterflies')
axs[-1].set_xlabel('datetime')
fig.tight_layout()
fig.savefig('../figs/time-evol_single-traps_scatter.pdf')
# %%
# converting data frame from wide to long format
# https://stackoverflow.com/questions/44941082/plot-multiple-columns-of-pandas-dataframe-using-seaborn
df_sns = df_upd.reset_index().melt(
    id_vars=['datetime', 'Cloud', 'Temp', 'Wind'],
    value_vars=cols,
    var_name='trap',
    value_name='total',
)
df_sns_sorted = df_sns.sort_values(by='datetime', axis='index')
#df_sns_sorted.to_csv(r'../data/updated_butterfly_data_long-format.csv')
df_sns
#%%
g = sns.relplot(x='datetime', y='total', hue='trap', style='trap', kind='scatter', data=df_sns)
g.figure.autofmt_xdate()
# %%
g = sns.PairGrid(df_sns, vars=['total','trap'], hue='Wind')
g.map(sns.scatterplot)
g.add_legend()
# %%
g = sns.catplot(x='trap', y='total', hue='Wind', data=df_sns)
g.fig.savefig('../figs/trap-vs-wind.pdf')
# %%
#sns.catplot(x='Wind', y='total', hue='trap', data=df_sns)
#sns.catplot(x='trap', y='total', kind='box', data=df_sns)
# %%
# there's a temperature way too high!
# %%
# computes pearson correlation coefficient
cols_corr = ['L-1','L-3','L-4','L-5','L-8','Temp']
corr = sns.heatmap(df_upd[cols_corr].corr(), annot=True)
# %%
