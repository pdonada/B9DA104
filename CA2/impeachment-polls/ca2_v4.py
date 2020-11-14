
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


# set directory
os.getcwd()
os.chdir('C:/github/B9DA104/CA2/impeachment-polls/')

################################################################

# read file
data_set = pd.read_csv('impeachment-polls.csv')
type(data_set)

# inspect data
data_set.head()
data_set.describe()
data_set.dtypes
print(data_set.shape)
data_set.info() #count non-null values in the columns
data_set.isnull().sum(axis=0) #count blanks in the columns

# subset with necessary columns only
#data_sub = data_set.drop(['Start','End','SampleSize','Category','Include?','Pollster','Sponsor','Pop','tracking','Text','URL','Notes'], axis = 1)
data_sub = data_set[['Yes','No','Unsure','Ind Sample']]
data_sub = data_sub.rename(columns={'Ind Sample':'IndSample'})

# imput values for NaN
data_sub.Unsure.fillna(value = 0, inplace = True) #here by zero as it just occurs when there was no real value
data_sub.IndSample.fillna(data_sub.IndSample.mean(), inplace = True) #here by mean as there should be always value for reference

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

# plot
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


# rescaling
array=data_sub.values
x=array[:,:]
scale=MinMaxScaler(feature_range=(0,2))
rescaled_data=scale.fit_transform(x)
np.set_printoptions(precision=2)
print(rescaled_data[0:5,:])

# linear regression
# use only one feature
x_data = data_sub.No

# split the data into training/testing sets
x_data_train = x[:-20]
x_data_test = x[-20:]

# split the targets into training/testing sets
y_data_train = data_sub.No[:-20]
y_data_test = data_sub.No[-20:]

# create linear regression object
lin_regres = LinearRegression()

# train the model using the training sets
lin_regres.fit(x_data_train, y_data_train)

# make predictions using the testing set
y_data_pred = lin_regres.predict(x_data_test)
print(y_data_pred)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))



 