# %%
'''
read butterfly data and explore dataset

TODO:

'''
# necessary libraries
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(
    style="darkgrid",
    color_codes=True,
    palette='Dark2',
)
#plt.style.use('bmh')
# %%
# DEFINE FUNCTIONS
def highlight_years(ax):
    for year in range(1982,2022,2):
        ax.axvspan(f'{year}-01-01', f'{year}-12-31', facecolor='black', edgecolor='none', alpha=0.1)

def read_and_plot(filepath, species, save_output=False):
    '''
    importing the Kaliningrad butterfly dataset into pandas dataframes
    and visualize the total number of insects per trap in daily resolution
    '''
    # IMPORT SINGLE SHEET
    df = pd.read_excel(
        io=filepath, 
        sheet_name=species,
        header=[0,1],                   # row to use for column labels
        skipfooter=2,
        dtype={('Year', 'Date'): str}   # save date as string, otherwise float abbreviation (01.10 -> 1.1)
    )
    # DATA PREPARATION
    df_upd = df.set_index(('Year', 'Date')).drop(labels='Total', axis=1).stack(level=0, dropna=False) # .dropna(axis=1, how='all')
    # labelling index columns and convert index to column
    df_upd.index.set_names('Day', level=0, inplace=True)
    df_upd.index.set_names('Year', level=1, inplace=True)
    df_upd = df_upd.reset_index()
    #print(df_upd)
    # convert date & year to datetime object
    df_upd['datetime'] = pd.to_datetime(
        arg = df_upd[['Year', 'Day']].astype(str).apply(' '.join, axis=1),
        format = '%Y %d.%m'
    )
    # and set datetime as index, then sort
    df_upd = df_upd.set_index(keys='datetime').sort_index(axis=0)

    # check data frame
    #print(df_upd)

    # PLOTTING
    # some species weren't caught by all traps
    cols = []
    possible_cols = ['L-1','L-2','L-3','L-4','L-5','L-6','L-7','L-8','L-9','L-10']
    for col in possible_cols:
        if col in df_upd.columns:
            cols.append(col)

    
    # using matplotlib subplots and pandas plotting capabilities
    fig, axs = plt.subplots(len(cols), 1, figsize=(12,9), sharex=True)
    df_upd[cols].plot(subplots=True, ax=axs)
    axs[3].set_ylabel('counts')
    fig.suptitle(species)
    for i in range(len(axs)):
        highlight_years(axs[i])
        axs[i].grid(visible=False, axis='x')
        #axs[i].set_yscale('log')
        axs[i].set_xlim(left='1982-01-01',right='2020-12-31') # OPTIONAL!
    fig.tight_layout()

    if save_output:
        fig.savefig(f'../figs/species_kaliningrad_time-evol_single-traps/{species}.pdf')
        # RETURN AND SAVE DATAFRAME OF SINGLE SPECIES
        df_upd.to_csv(f'../data/species_kaliningrad/{species}.csv')
    
    return df_upd

# some species weren't caught by all traps
def which_cols(df):
    cols = []
    possible_cols = ['L-1','L-2','L-3','L-4','L-5','L-6','L-7','L-8','L-9','L-10']
    for col in possible_cols:
        if col in df.columns:
            cols.append(col)
    return cols
# %%
# INPUT
path = r'C:\Users\hotte\Desktop\Research\data\Lepidoptera_records_Courish_Spit_no_gender.xls'
sheets = ['Vanessa atalanta', 'Vanessa cardui', 'Inachis io', 'Issoria lathonia', 'Aglais urticae',
    'Aporia crataegi', 'Apatura ilia', 'Aphantopus hyperanthus', 'Araschnia levana', 'Nymphalis antiopa',
    'Nymphalis polychloros', 'Nymphalis xanthomelas', 'Papilio machaon', 'Polygonia c-album', 'Pararge aegeria'
]

# import dataset into dictionary, holding individual excel sheets
#################################################################
species = dict()
save_output_flag = False
for spec in sheets:
    df = read_and_plot(path, spec, save_output_flag)
    species[f'{spec}'] = df
print('--- done ---')
#################################################################
# %%
# converting data frame from wide to long format
# https://stackoverflow.com/questions/44941082/plot-multiple-columns-of-pandas-dataframe-using-seaborn

species_long = dict()
for i, s in enumerate(sheets):
    df = species[s]
    species_long[s] = df.reset_index().melt(
        id_vars=['datetime', 'Day', 'Year', 'Cloud', 'Temp', 'Wind'],
        value_vars=which_cols(df),
        var_name='trap',
        value_name='total',
    )

species_long[sheets[0]]
#%%
# sum up individual traps
species_summed = dict()
for i, s in enumerate(sheets):
    df = species[s]
    df['count'] = df[which_cols(df)].sum(axis=1, min_count=1)
    species_summed[s] = df.reset_index()

species_summed[sheets[0]]
#%% 
# SAVING DICTIONARIES
import pickle
for dic, dic_str in zip([species, species_long, species_summed], ['species', 'species_long', 'species_summed']):
    with open('../data/'+dic_str+'.pkl', 'wb') as file:
        pickle.dump(dic, file)
'''
with open('species.pkl', 'rb') as file:
    species = pickle.load(file)
'''
#%%
# compute total number of counts per species
total = np.zeros(len(sheets))
for i, s in enumerate(sheets):
    df = species[s]
    cols = which_cols(df)
    total[i] = df[cols].sum().sum()

# and plotting
fig, ax = plt.subplots()
sns.barplot(x=sheets, y=total)
plt.xticks(rotation=45, ha='right')
ax.set_ylabel('counts')
#ax.set(yscale='log')
fig.suptitle('total abundance of species')
fig.tight_layout()
fig.savefig('../figs/total-count_per_species.pdf')
# %%
# first and last day of appearance per species
def first_last_day(df):
    '''
    data frame must have a 'Year' column
    '''
    years = np.arange(1982, 2020+1)
    first = np.zeros(len(years))
    last = np.zeros(len(years))
    for i in range(len(years)):
        try:
            # select first index with butterfly count greater than 0
            xfirst = df[(df['Year'] == years[i]) & (df['count'] >= 1)].index[0]
            # and last index
            xlast = df[(df['Year'] == years[i]) & (df['count'] >= 1)].index[-1]
        except:
            print(f'No counts in {years[i]}.')
            first[i], last[i] = np.nan, np.nan
        else:
            first[i], last[i] = (df['datetime'].iloc[[xfirst, xlast]] - pd.Timestamp(f'1/1/{years[i]}')).values.astype('timedelta64[D]')
    return first, last

# %%
years = np.arange(1982, 2020+1) # years as integers

# compute total number of counts per year per species
total_year = dict()
for s in sheets:
    df = species_summed[s]
    dummy_counts = np.zeros(len(years))
    for i in range(len(dummy_counts)):
        dummy_counts[i] = df['count'][df['Year'] == years[i]].sum(min_count=1)
    total_year[s] = dummy_counts
# %%
# compute first and last day of sight throughout entire time series per species
# NOTE: by now, seaborn can't perform WEIGHTED linear regression, compare: https://github.com/mwaskom/seaborn/issues/2163
#       however, in other libraries (e.g., numpy, scipy, scikit-learn) weights can be added!
years_dt = pd.date_range(start='1/1/1982',end='1/1/2020', freq='YS').values.astype('datetime64[D]') # years as datetime object
first, last = dict(), dict()
for s in sheets:
    df = species_summed[s]
    first[s], last[s] = first_last_day(df)
# and plotting...
    f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]}, figsize=(8, 6))
    axs[0].bar(years, total_year[s], color='grey')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('total count')
    sns.regplot(x=years, y=last[s], ax=axs[1])
    sns.regplot(x=years, y=first[s], ax=axs[1])
    axs[1].set_ylabel(r'days since Jan 1$^\mathregular{st}$')
    axs[1].set_xlabel('year')
    axs[1].legend(labels=['last', 'first'], loc='center right')
    f.suptitle(f'{s}')
    f.tight_layout()
    f.savefig(f'../figs/species_kaliningrad_first-last-day/{s}.pdf')

# alternatively...

#from matplotlib import gridspec
#gs = gridspec.GridSpec(2, 1, height_ratios=[1,3])
#ax0 = plt.subplot(gs[0])
#ax1 = plt.subplot(gs[1])
#%%
from datetime import datetime
def number_of_day_to_date(day_num):
    print('Day number:',day_num)
    res = datetime.strptime('2020-'+str(day_num), "%Y-%j").strftime("%d-%m-%Y")
    print('Resolved date:',res)

number_of_day_to_date(100)
# %%
# transform insect counts to weekly data / change temporal resolution
species_weekly = dict()
cols = ['datetime', 'year', 'count']
# loop over species
for s in sheets:
    df = species_summed[s]
    # create empty data frame to hold weekly counts
    df_temp = pd.DataFrame(columns=cols)
    # loop over years
    for y in years:
        # extract data from single year
        df_y = df[df.Year == y]
        count_temp = []
        year_temp = []
        datetime_temp = []
        # create index [0, 7, 14, ...] to sum up counts of 7 consecutive days,
        # append weekly count, year and datetime of first (!) day to list
        for i in range(0, len(df_y.index-7), 7):
            count_temp.append(df_y['count'].iloc[i:i+7].sum(min_count=1))
            year_temp.append(df_y['Year'].iloc[i])
            datetime_temp.append(df_y['datetime'].iloc[i])
        # update output data frame
        zipped = list(zip(datetime_temp,year_temp,count_temp))
        df_temp = df_temp.append(pd.DataFrame(zipped, columns=cols), ignore_index=True)
    # add data frame holding weekly counts to species dict
    species_weekly[s] = df_temp
# %%
# plot time series of weekly count of atalanta & cardui (unedited and cropped x-axis)
import itertools
# set palette 
palette = itertools.cycle(sns.color_palette())

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,4))
butterflies = ['Vanessa atalanta', 'Vanessa cardui']
for i, s in enumerate(butterflies):
    # color
    c = next(palette)
    df = species_weekly[s]
    axs[i].plot(df['datetime'], df['count'], c=c)
    axs[i].legend(labels=[s], loc='upper left')
    #axs[i].set_yscale('log')
    highlight_years(axs[i])
    axs[i].set_ylabel('count')
    # cropping x-axis to 1982-2020
    axs[i].set_xlim(left=pd.to_datetime('1982-01-01'),right=pd.to_datetime('2020-12-31')) # OPTIONAL!
    #axs[i].set_ylim(bottom=0, top=400)
axs[1].set_xlabel('year')
fig.suptitle('Weekly abundance of V. atalanta & V. cardui')
fig.tight_layout()
fig.savefig('../figs/atalanta-vs-cardui/time-evol-weekly_atalanta-vs-cardui.pdf')
#%%
# CREATE BAR PLOT OF YEARLY ABUNDANCE OF V. ATALANTA & V. CARDUI
'''
seaborn    : doesn't sum up values belonging to same bar automatically, but rather computes average & its error
             requires data in "long"-format
             adds, by default, label to x-axis for every bar
pandas     : requires proper data format
matplotlib : everything's possible, but fairly low-level

'''
# compute yearly abundance
dic = species_weekly
va_vc_ = dic[butterflies[0]].merge(
    right=dic[butterflies[1]],
    on=['datetime', 'year'],
    suffixes=('_va', '_vc')
).drop(labels='datetime', axis=1)
#va_vc_['year'] = pd.to_datetime(va_vc_['year'], format='%Y')
va_vc_ = va_vc_.groupby(['year']).sum(min_count=1)
va_vc_
# %%
# SEABORN!
df = va_vc_.reset_index().melt(
    id_vars=['year'],
    value_vars=['count_va', 'count_vc'],
    var_name='species',
    value_name='count'
)
fig, ax = plt.subplots(figsize=(10,6))
g = sns.barplot(data=df, x='year', y='count', hue='species', ax=ax)
ax.set_xticklabels([str(year) for year in range(1982, 2021, 1)], rotation=90)
#ticks = ax.get_xticks()
#ax.set_xticks(ticks[::2])
#fig.autofmt_xdate()

# replace labels of legend
# https://stackoverflow.com/questions/45201514/how-to-edit-a-seaborn-legend-title-and-labels-for-figure-level-functions
for t, l in zip(g.legend_.texts, butterflies):
    t.set_text(l)

fig.suptitle('Yearly abundance of V. atalanta & V. cardui')
fig.tight_layout()
fig.savefig('../figs/atalanta-vs-cardui/time-evol-yearly_atalanta-vs-cardui.pdf')
# %%
# MATPLOTLIB!
w = 0.4
df = va_vc_
x = df.index
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x-w/2, df['count_va'], width=w)
ax.bar(x+w/2, df['count_vc'], width=w)
ax.set_ylim(top=2000)
ax.set_xticks(x+w/4)
ax.set_xticklabels(x, rotation=90)
ax.legend(labels=butterflies)
fig.tight_layout()
# ...
#%%
# compute yearly count for all species in single, long-format dataframe
df_species_yearly = pd.DataFrame()
for i, s in enumerate(sheets):
    df_temp = species_weekly[s]
    df_temp = df_temp.groupby('year', as_index=False).sum(min_count=1)
    df_temp['species'] = s
    df_species_yearly = pd.concat([df_species_yearly, df_temp])
df_species_yearly = df_species_yearly.reset_index(drop=True)
df_species_yearly
# %%
data = df_species_yearly.groupby('year', as_index=False).sum(min_count=1)
fig, ax = plt.subplots(figsize=(8,6))
sns.regplot(
    data=data, x='year', y='count', ax=ax,
)
fig.suptitle('Total abundance of insects')
fig.tight_layout()
fig.savefig('../figs/total-count_per_year.pdf')
# %%
