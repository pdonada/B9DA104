import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#pip list #Pandas 0.24.2 has an issue with fit_scores() method
#pip install pandas==0.23.4



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
data_sub = data_set[['Yes', 'No', 'Rep Yes', 'Rep No', 'Dem Yes', 'Dem No']]

# rename columns to same standard
data_sub = data_sub.rename(columns={'Rep Yes':'RepYes', 'Rep No':'RepNo', 'Dem Yes':'DemYes', 'Dem No':'DemNo'})

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
core_p = ('Correlation: ', data_sub['Yes'].corr(data_sub['RepYes']))
print(*core_p)

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

################################################################################
# data imputation on DemYes columns using Linear Regression
# checking correlation between Yes and RepYes to see if can be used for LR
print('Correlation: ', data_sub['Yes'].corr(data_sub['DemYes']))

# to split training and testing subsests will make a filter to avoid NaN values on RepYes
df_filter = data_sub[data_sub['DemYes']>=0].copy()
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
    y_train = np.array(df_train['DemYes']).reshape(-1,1)
    x_test = np.array(df_test['Yes']).reshape(-1,1)
    y_test = np.array(df_test['DemYes']).reshape(-1,1)
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
df_missing = data_sub[data_sub['DemYes'].isnull()].copy()   

# predicting NaN values on RepYes with the model for NaN values
x_test_lr = np.array(df_missing['Yes']).reshape(-1,1)

x_train_lr = np.array(df_filter['Yes']).reshape(-1,1)
y_train_lr = np.array(df_filter['DemYes']).reshape(-1,1)

# creating LR object
model_lr = LinearRegression()

# fiting LR model
model_lr.fit(x_train_lr, y_train_lr)
print('Linear regression predictions: ', model_lr.predict(x_test_lr)[0])

# store prediction result in a variable
pred = model_lr.predict(x_test_lr)
print(pred)

# crating variable with RepYes values to imput predicted values in NaN psitions
repyes_vals = data_sub['DemYes'].values
print(repyes_vals)

# loop to imput values on RepYes NaN with predicted values
i_value = 0
for i in range(len(repyes_vals)):
    if np.isnan(repyes_vals[i]):
        repyes_vals[i] = pred[i_value]
        i_value += 1

# copiando values from variable to RepYes column
data_sub['DemYes'] = repyes_vals           

# plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, pred, color='blue', linewidth=3)

################################################################################
# Multivariate Linear Regression to predict Yes

# creating different dataframe with columns to be used as target and variables
df = pd.DataFrame(data_sub, columns=['Yes','RepYes','DemYes'])

# creating 'x' and 'y' objects to be used into the Multiple Linear Regression
x = pd.DataFrame(df, columns=['RepYes','DemYes'])
y = pd.DataFrame(df, columns=['Yes'])

# ploting to see linear relationship 
plt.scatter(df['DemYes'], df['Yes'], color='red')
plt.xlabel('DemYes', fontsize=14)
plt.ylabel('Yes', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['RepYes'], df['Yes'], color='green')
plt.xlabel('RepYes', fontsize=14)
plt.ylabel('Yes', fontsize=14)
plt.grid(True)
plt.show()

# spliting database
k2 = int(len(df['Yes']) * 0.2) # 20% samples
k8 = int(len(df['Yes']) * -0.8) # 80% samples

x_train = x[k8:] #80%
x_train = np.c_[np.ones(len(x_train),dtype='int64'),x_train]
y_train = y[k8:]

x_test = x[:k2] #20%
x_test = np.c_[np.ones(len(x_test),dtype='int64'),x_test]
y_test = y[:k2]
y_test = y_test.round(0)

# creating LR object
model_mr = LinearRegression()

# fiting LR model
model_mr.fit(x_train, y_train)
print('multi linear regression predictions: ', model_mr.predict(x_test)[0])


# store prediction result in a variable
pred = model_mr.predict(x_test)
pred = pred.round(0)

# verifying accuracy
# coefficients
print('Coefficients: \n', model_mr.coef_)
# mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))
# variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, pred))


# prepare model data point for visualization
pred_f = pred.flatten()
pred_df = pd.DataFrame({'Predict_Yes':pred_f})

dfk = pd.concat([y_test, pred_df], axis=1)

error = dfk['Yes'] - dfk['Predict_Yes']

# plot outputs

  

