# This module contains all functions used multiple times throughout the 
# data analysis.

################################################################################

def highlight_years(ax):
    '''
    highlighting individual years by plotting vertical bars
    '''
    for year in range(1982,2022,2):
        ax.axvspan(f'{year}-01-01', f'{year}-12-31', facecolor='black', 
                   edgecolor='none', alpha=0.1)

################################################################################

def which_cols(df):
    # some species weren't caught by all traps -> check
    cols = []
    possible_cols = ['L-1','L-2','L-3','L-4','L-5','L-6','L-7','L-8','L-9','L-10']
    for col in possible_cols:
        if col in df.columns:
            cols.append(col)
    return cols

################################################################################

def import_pickled_data():
    import pickle
    from vals import pkl_files
    dummy = []
    for file in pkl_files:
        with open('../../data/'+file, 'rb') as f:
            dummy.append(pickle.load(f))
    return dummy

################################################################################

from datetime import datetime
def number_of_day_to_date(day_num):
    print('Day number:',day_num)
    res = datetime.strptime('2020-'+str(day_num), "%Y-%j").strftime("%d-%m-%Y")
    print('Resolved date:',res)

################################################################################

import numpy as np
import pandas as pd
def first_last_day(df):
    '''
    data frame must have a 'Year' column!
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