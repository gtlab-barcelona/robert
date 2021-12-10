# %%
'''
read butterfly data and explore dataset

TODO:

'''
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid", palette='Dark2')
#plt.style.use('bmh')
#%%
# DEFINE FUNCTIONS
def highlight_years(ax):
        for year in range(1982,2022,2):
            ax.axvspan(f'{year}-01-01', f'{year}-12-31', facecolor='black', edgecolor='none', alpha=0.1)

def read_and_plot(filepath, species):
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
    df_upd = df.set_index(('Year', 'Date')).drop(labels='Total', axis=1).dropna(axis=1, how='all').stack(0)
    # labelling index columns and convert index to column
    df_upd.index.set_names('Day', level=0, inplace=True)
    df_upd.index.set_names('Year', level=1, inplace=True)
    df_upd = df_upd.reset_index()
    # convert date & year to datetime object
    df_upd['datetime'] = pd.to_datetime(
        arg = df_upd[['Year', 'Day']].astype(str).apply(' '.join, axis=1),
        format = '%Y %d.%m'
    )
    # and set datetime as index, then sort
    df_upd = df_upd.set_index(keys='datetime').sort_index(axis=0)

    # PLOTTING
    # some species weren't caught by all traps
    cols = []
    possible_cols = ['L-1','L-2','L-3','L-4','L-5','L-6','L-7','L-8','L-9','L-10']
    for col in possible_cols:
        if col in df_upd.columns:
            cols.append(col)
    
    #print(df_upd)

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
    fig.savefig(f'../figs/species_kaliningrad_time-evol_single-traps/{species}.pdf')
    
    # RETURN AND SAVE DATAFRAME OF SINGLE SPECIES
    df_upd.to_csv(f'../data/species_kaliningrad/{species}.csv')
    return df_upd
# %%
# INPUT
path = r'C:\Users\hotte\Desktop\Research\data\Lepidoptera_records_Courish_Spit_no_gender.xls'
sheets = ['Vanessa atalanta', 'Vanessa cardui', 'Inachis io', 'Issoria lathonia', 'Aglais urticae',
    'Aporia crataegi', 'Apatura ilia', 'Aphantopus hyperanthus', 'Araschnia levana', 'Nymphalis antiopa',
    'Nymphalis polychloros', 'Nymphalis xanthomelas', 'Papilio machaon', 'Polygonia c-album', 'Pararge aegeria'
]

##################################
species = dict()
for spec in sheets:
    df = read_and_plot(path, spec)
    species[f'{spec}'] = df
print('--- done ---')
##################################
# %%
# converting data frame from wide to long format
# https://stackoverflow.com/questions/44941082/plot-multiple-columns-of-pandas-dataframe-using-seaborn

# pd.DataFrame.from_dict(species)
# %%
# some species weren't caught by all traps
def which_cols(df):
    cols = []
    possible_cols = ['L-1','L-2','L-3','L-4','L-5','L-6','L-7','L-8','L-9','L-10']
    for col in possible_cols:
        if col in df.columns:
            cols.append(col)
    return cols

#%%
# compute total number of counts per species
total = np.zeros(len(sheets))
for i, s in enumerate(sheets):
    df = species[s]
    cols = which_cols(df)
    total[i] = df[cols].sum().sum()
#%%
fig, ax = plt.subplots()
sns.barplot(x=sheets, y=total)
plt.xticks(rotation=45, ha='right')
ax.set_ylabel('counts')
#ax.set(yscale='log')
fig.suptitle('total abundance of species')
fig.tight_layout()
fig.savefig('../figs/total-count_per_species.pdf')
# %%
