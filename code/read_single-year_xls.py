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
#plt.style.use('bmh')
# %%
path = r'C:\Users\hotte\Desktop\Research\data\Lepidoptera_records_Courish_Spit_no_gender.xls'

# importing single year
butterfly_df = pd.read_excel(
    io=path,
    sheet_name='Vanessa cardui',
    usecols="A,BI,BJ",  # for now, just use first year
    header=1,       # row to use for column labels
    skipfooter=2,
    converters={'Date': str}    # save date as string to change it easier
    #index_col=0,   # make date the index of data frame
    #skiprows=1,
)
print(type(butterfly_df['L-4.9'][0]))
#%%

# adding year to date column
butterfly_df['Date'] = butterfly_df['Date'] + '.1982'

# convert date to datetime object
butterfly_df['Date'] = pd.to_datetime(
    arg=butterfly_df['Date'],
    format='%d.%m.%Y',
)

# set date column as index
butterfly_df.set_index('Date', inplace=True)

# check data frame
butterfly_df
# %%
# plotting
#cols = ['L-1','L-2','L-3','L-4','L-5']
cols = ['L-4.9', 'L-5.9']

#fig, axs = plt.subplots(figsize=(12,4))
#butterfly_df[['L-1','L-2','L-3','L-4','L-5']].plot.area(ax=axs)
axs = butterfly_df[cols].plot(
    figsize=(12,8), 
    subplots=True,
    #ylabel='number of v. atalanta'
)
plt.show()
# use seaborn!
# %%
# some more plotting
butterfly_df[cols].plot()
# %%
