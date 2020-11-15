
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# set directory
os.getcwd()
os.chdir('C:/github/B9DA104/CA2/impeachment-polls/')

################################################################

# read file
data_set = pd.read_csv('impeachment-polls.csv')

# inspect data
type(data_set)
data_set.head()
data_set.describe()
data_set.dtypes
print(data_set.shape)
data_set.info() #count non-null values in the columns
data_set.isnull().sum(axis=0) #count blanks in the columns

# subset with necessary columns only
data_sub = data_set[['Yes','No','Unsure','Rep Yes', 'Rep No']]

# rename columns to same standard
data_sub = data_sub.rename(columns={'Rep Yes':'RepYes', 'Rep No':'RepNo'})

# imput zeros on NaN values for Unsure
data_sub.Unsure.fillna(value = 0, inplace = True)

# inspect data
data_sub.head()
data_sub.describe()
data_sub.dtypes
print(data_sub.shape)
data_sub.info() #count non-null values in the columns
data_sub.isnull().sum(axis=0) #count blanks in the columns

# correlations
data_sub.corr()

# skew
data_sub.skew()

# histogram
hist = data_sub.hist(bins=20)

# box plot
box = data_sub.plot(kind='box', subplots=True, layout=(4,3),sharex=False,sharey=False)
box1= data_sub.plot(kind='box', subplots=False, layout=(4,3), sharex=False, sharey=False)

# density plot
dens = data_sub.plot(kind='density', subplots=True, layout=(4,3), sharex=False, sharey=False)

# correlation heat map
corr = data_sub.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels = corr.columns.values)

# pandas scatter_matrix
ocor_a = pd.plotting.scatter_matrix(data_sub, alpha = 0.2)
for i in ocor_a.flatten():
    i.xaxis.label.set_rotation(90)
    i.yaxis.label.set_rotation(0)
    i.yaxis.label.set_ha('right')


# data imputation on RepYes columns using Linear Regression
# checking correlation between Yes and RepYes to see if can be used for LR
print('Correlation: ', data_sub['Yes'].corr(data_sub['RepYes']))

# to split training and testing subsests will make a filter to avoid NaN values on RepYes
df_filter = data_sub[data_sub['RepYes']>=0].copy()
print(df_filter.shape)

# creating objects to house predictions and real values
y_pred = []
y_true = []


# using k-fold validation for the model using 10 folds.
kf = KFold(n_splits=10, random_state = 42)
for train_index, test_index in kf.split(df_filter):
    df_test = df_filter.iloc[test_index]
    df_train = df_filter.iloc[train_index]
# defining input and output.    
    x_train = np.array(df_train['Yes']).reshape(-1,1)
    y_train = np.array(df_train['RepYes']).reshape(-1,1)
    x_test = np.array(df_test['Yes']).reshape(-1,1)
    y_test = np.array(df_test['RepYes']).reshape(-1,1)
# fiting LR model.  
    model = LinearRegression()
    model.fit(x_train, y_train)
# generating/appending prediction values to the objects created before    
    y_pred.append(model.predict(x_test)[0])
    y_true.append(y_test[0])  

# checking performance of model with mean square error
print(len(y_pred))
print(len(y_true))
print('Mean Square Error: ', mean_squared_error(y_true, y_pred))

# creating list with NaN values on RepYes to be used in the model
df_missing = data_sub[data_sub['RepYes'].isnull()].copy()   

# predicting NaN values on RepYes with the model for NaN values
x_test_lr = np.array(df_missing['Yes']).reshape(-1,1)

x_train_lr = np.array(df_filter['Yes']).reshape(-1,1)
y_train_lr = np.array(df_filter['RepYes']).reshape(-1,1)

# creating LR object
model_lr = LinearRegression()

# fiting LR model
model_lr.fit(x_train_lr, y_train_lr)
print('Linear regression predictions: ', model_lr.predict(x_test_lr)[0])

# store prediction result in a variable
pred = model_lr.predict(x_test_lr)
print(pred)

# crating variable with RepYes values to imput predicted values in NaN psitions
repyes_vals = data_sub['RepYes'].values
print(repyes_vals)

# loop to imput values on RepYes NaN with predicted values
i_value = 0
for i in range(len(repyes_vals)):
    if np.isnan(repyes_vals[i]):
        repyes_vals[i] = pred[i_value]
        i_value += 1

# copiando values from variable to RepYes column
data_sub['RepYes'] = repyes_vals           

# plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, pred, color='blue', linewidth=3)
print(x_test)




##############################################################################

data_set.head()
print('Correlation: ', data_sub['Yes'].corr(data_sub['RepYes']))

df_filter = data_sub[data_sub['RepYes']>=0].copy()
print(df_filter.shape)
y_pred = []
y_true = []

kf = KFold(n_splits=10, random_state = 42)
for train_index, test_index in kf.split(df_filter):
    df_test = df_filter.iloc[test_index]
    df_train = df_filter.iloc[train_index]

    x_train = np.array(df_train['Yes']).reshape(-1,1)
    y_train = np.array(df_train['RepYes']).reshape(-1,1)
    x_test = np.array(df_test['Yes']).reshape(-1,1)
    y_test = np.array(df_test['RepYes']).reshape(-1,1)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    y_pred.append(model.predict(x_test)[0])
    y_true.append(y_test[0])    
    
print(len(y_pred))
print(len(y_true))
print('Mean Square Error: ', mean_squared_error(y_true, y_pred))


df_missing = data_sub[data_sub['RepYes'].isnull()].copy()   
print(df_missing) 
print(len(df_missing))

x_test_lr = np.array(df_missing['Yes']).reshape(-1,1)

x_train_lr = np.array(df_filter['Yes']).reshape(-1,1)
y_train_lr = np.array(df_filter['RepYes']).reshape(-1,1)



model_lr = LinearRegression()
model_lr.fit(x_train_lr, y_train_lr)
print('Linear regression predictions: ', model_lr.predict(x_test_lr)[0])
pred = model_lr.predict(x_test_lr)
print(pred)


data_sub.RepYes.fillna(value = 0, inplace = True)

print(data_sub['RepYes'])
print(data_sub['RepYes'].isnull())

repyes_vals = data_sub['RepYes'].values
print(repyes_vals)

i_value = 0
for i in range(len(repyes_vals)):
    if np.isnan(repyes_vals[i]):
        repyes_vals[i] = pred[i_value]
        i_value += 1
        
print(repyes_vals)

data_sub['RepYes'] = repyes_vals        
        
