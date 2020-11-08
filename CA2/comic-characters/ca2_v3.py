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
ds_sub.head()
ds_sub.describe(include = 'all')
ds_sub.info() #count non-null values in the columns
ds_sub.isnull().sum(axis=0) #count blanks in the columns

ds_sub = ds_sub.dropna(subset=['Year'])
ds_sub = ds_sub.dropna(subset=['SEX'])
ds_sub = ds_sub.dropna(subset=['ALIGN'])
ds_sub = ds_sub.dropna(subset=['APPEARANCES'])

# change columns object type
ds_sub = ds_sub.astype({'Year': int, 'SEX':str, 'ALIGN':str, 'APPEARANCES': int})

# creating binary column based on 'SEX'
ds_sub['SEX'].nunique()
ds_sub['SEX'].unique()

sex_cond = [
        (ds_sub['SEX']=='Male Characters'),
        (ds_sub['SEX']=='Female Characters'),
        (ds_sub['SEX']=='Genderfluid Characters'),
        (ds_sub['SEX']=='Agender Characters')
        ] # conditions to test sex

sex_val = [0, 1, 2, 3] # values to be attributed

ds_sub['SEX_B'] = np.select(sex_cond, sex_val) #create new column 

# creating binary column based on 'ALIGN'
ds_sub['ALIGN'].nunique()
ds_sub['ALIGN'].unique()

align_cond = [
        (ds_sub['ALIGN']=='Good Characters'),
        (ds_sub['ALIGN']=='Neutral Characters'),
        (ds_sub['ALIGN']=='Bad Characters')
        ] # conditions to test sex

align_val = [0, 1, 2] # values to be attributed

ds_sub['ALIGN_B'] = np.select(align_cond, align_val) #create new column 


histogram = ds_sub.hist()


