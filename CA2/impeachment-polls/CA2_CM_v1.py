import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# set directory
os.getcwd()
os.chdir('C:/github/B9DA104/CA2/')

# read file
data_set = pd.read_csv('raw-responses-1.csv')
data_set = data_set.dropna()

# inspect data
type(data_set)
data_set.head()
data_set.describe()
data_set.dtypes
print(data_set.shape)
data_set.info() #count non-null values in the columns
data_set.isnull().sum(axis=0) #count blanks in the columns

# subset with necessary columns only
columns_or = ['q0001','q0002','q0005','q0008_0001','q0008_0002','q0008_0003','q0008_0004','q0008_0005'
              ,'q0008_0006','q0008_0007','q0008_0008','q0008_0009','q0008_0010','q0008_0011','q0008_0012'
              ,'q0017','q0018','q0022','q0024','q0026','age3','q0028','q0029' ]

columns_cat = ['q0001_cat','q0002_cat','q0005_cat','q0008_0001_cat','q0008_0002_cat','q0008_0003_cat'
               ,'q0008_0004_cat','q0008_0005_cat','q0008_0006_cat','q0008_0007_cat','q0008_0008_cat'
               ,'q0008_0009_cat','q0008_0010_cat','q0008_0011_cat','q0008_0012_cat','q0017_cat','q0018_cat'
               ,'q0022_cat','q0024_cat','q0026_cat','age3_cat','q0028_cat','q0029_cat']


data_sub = data_set[columns_or]

data_sub.dtypes
data_sub.info() #count non-null values in the columns
data_sub.isnull().sum(axis=0) #count blanks in the columns

# categorical encoding (text data into numerical)
obj_df = data_sub

# change all columns to 'category' type to apply categorical transformation
for col in [columns_or]:
    obj_df[col] = obj_df[col].astype('category')

# creating categorical columns based in each df column
obj_df['q0001_cat'] = obj_df['q0001'].cat.codes
obj_df['q0002_cat'] = obj_df['q0002'].cat.codes
obj_df['q0005_cat'] = obj_df['q0005'].cat.codes
obj_df['q0008_0001_cat'] = obj_df['q0008_0001'].cat.codes
obj_df['q0008_0002_cat'] = obj_df['q0008_0002'].cat.codes
obj_df['q0008_0003_cat'] = obj_df['q0008_0003'].cat.codes
obj_df['q0008_0004_cat'] = obj_df['q0008_0004'].cat.codes
obj_df['q0008_0005_cat'] = obj_df['q0008_0005'].cat.codes
obj_df['q0008_0006_cat'] = obj_df['q0008_0006'].cat.codes
obj_df['q0008_0007_cat'] = obj_df['q0008_0007'].cat.codes
obj_df['q0008_0008_cat'] = obj_df['q0008_0008'].cat.codes
obj_df['q0008_0009_cat'] = obj_df['q0008_0009'].cat.codes
obj_df['q0008_0010_cat'] = obj_df['q0008_0010'].cat.codes
obj_df['q0008_0011_cat'] = obj_df['q0008_0011'].cat.codes
obj_df['q0008_0012_cat'] = obj_df['q0008_0012'].cat.codes
obj_df['q0017_cat'] = obj_df['q0017'].cat.codes
obj_df['q0018_cat'] = obj_df['q0018'].cat.codes
obj_df['q0022_cat'] = obj_df['q0022'].cat.codes
obj_df['q0024_cat'] = obj_df['q0024'].cat.codes
obj_df['q0026_cat'] = obj_df['q0026'].cat.codes
obj_df['age3_cat'] = obj_df['age3'].cat.codes
obj_df['q0028_cat'] = obj_df['q0028'].cat.codes
obj_df['q0029_cat'] = obj_df['q0029'].cat.codes

# creating new df excluding original columns
df_cat = obj_df.drop([columns_or], axis=1)


# Gaussian Naive Bayes Classification

# creating 'x' and 'y' objects to be used into the Multiple Linear Regression
x = pd.DataFrame(df_cat, columns=['q0002_cat','q0005_cat','q0008_0001_cat','q0008_0002_cat','q0008_0003_cat'
               ,'q0008_0004_cat','q0008_0005_cat','q0008_0006_cat','q0008_0007_cat','q0008_0008_cat'
               ,'q0008_0009_cat','q0008_0010_cat','q0008_0011_cat','q0008_0012_cat','q0017_cat','q0018_cat'
               ,'q0022_cat','q0024_cat','q0026_cat','age3_cat','q0028_cat','q0029_cat'])

y = pd.DataFrame(df_cat, columns=['q0001_cat'])

# feature extraction
np.set_printoptions(suppress=True)

test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(x, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

# recriating x with better scored features
x = pd.DataFrame(df_cat, columns=['q0008_0001_cat','q0029_cat','q0008_0006_cat','q0022_cat','q0008_0009_cat'
                                  ,'q0026_cat','q0017_cat','q0008_0002_cat','q0008_0004_cat','q0008_0003_cat'
                                  ,'q0008_0007_cat','q0008_0008_cat','q0005_cat','q0002_cat','q0018_cat'])

# spliting database
k2 = int(len(df_cat['q0001_cat']) * 0.2) # 20% samples


x_train = x[k2:] #80%
x_train = np.c_[np.ones(len(x_train),dtype='int64'),x_train]
y_train = y[k2:]

x_test = x[:k2] #20%
x_test = np.c_[np.ones(len(x_test),dtype='int64'),x_test]
y_test = y[:k2]
y_test = y_test.round(0)

# train the model
clf = GaussianNB()
clf.fit(x_train, y_train)

# use the model to predict the labels of the test data
predicted = clf.predict(x_test)
expected = y_test

print(predicted) 
print(expected) 


predicted_df = pd.DataFrame({'q0001_cat':predicted})

predicted_m = predicted_df.values
exprected_m = expected.values

# check performance 
matches = (predicted_m == exprected_m)
correct = (matches.sum() / float(len(matches)))*100
print('Coredictions match(%): ',correct.round(2))

print(metrics.confusion_matrix(exprected_m, predicted_m))

