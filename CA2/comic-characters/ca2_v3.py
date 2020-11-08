import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


# set directory
os.getcwd()
os.chdir('C:/github/B9DA104/CA2/comic-characters/')

################################################################

# read file
data_set = pd.read_csv('marvel-wikia-data.csv')
type(data_set)

# inspect data
data_set.head()
data_set.describe()
data_set.dtypes
print(data_set.shape)

# subset the main dataframe and clean blank rows
ds_sub = data_set[['Year','SEX','ALIGN','APPEARANCES']]
type(ds_sub)

ds_sub = ds_sub.dropna(subset=['Year'])
ds_sub = ds_sub.dropna(subset=['SEX'])
ds_sub = ds_sub.dropna(subset=['ALIGN'])
ds_sub = ds_sub.dropna(subset=['APPEARANCES'])

ds_sub.head()
ds_sub.describe(include = 'all')
ds_sub['Year'].describe()
ds_sub.info() #count non-null values in the columns
ds_sub.isnull().sum(axis=0) #count blanks in the columns



histogram = ds_hist.hist(bins = 20)

# subset cleaning empty rows
ds_hist.dropna(subset = ['Year'], inplace=True)
