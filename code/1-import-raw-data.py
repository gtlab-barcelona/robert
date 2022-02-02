# %%
'''
import raw dataset,
plot daily count of individual traps for all species and
save processed data as pickle-files

TODO:

'''
# import libraries
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(
    style="darkgrid",
    color_codes=True,
    palette='Dark2',
)
# import (personal) functions
from funcs import highlight_years, which_cols
# %%
# DEFINE IMPORT AND PLOT FUNCTION
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
# %%
# RUN SCRIPT
path = r'..\data\Lepidoptera_records_Courish_Spit_no_gender.xls'
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