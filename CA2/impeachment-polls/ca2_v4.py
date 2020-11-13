
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import os

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
data_sub = data_set.drop(['Start','End','SampleSize','Category','Include?','Pollster','Sponsor','Pop','tracking','Text','URL','Notes'], axis = 1)


# update column 'Unsure' for zeros when blank
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

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()




data_set = data_set.dropna(subset=['YEAR'])
data_set = data_set.dropna(subset=['FEMALE_CHARACTERS'])
data_set = data_set.dropna(subset=['MALE_CHARACTERS'])


# change columns object type
data_set['Category'] = data_set['Category'].astype(str)

# creating columns for rate
data_set['FEMALE_RATE'] = round((data_set['FEMALE_CHARACTERS'] / data_set['TOTAL']).astype(float),3)

# ploting data for evaluation
sns.pairplot(data_set, hue = None, diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 3)
plt.suptitle('Character sex by Year', 
             size = 20)

sns.pairplot(
    data_set,
    x_vars=["MALE_CHARACTERS", "FEMALE_CHARACTERS",'FEMALE_RATE'],
    y_vars=["YEAR"],
)


# Checking for Linearity
plt.scatter(data_set['FEMALE_RATE'], data_set['YEAR'], color='red')
plt.title('Female Rate X Year', fontsize=14)
plt.xlabel('Female Rate', fontsize=14)
plt.ylabel('Year', fontsize=14)
plt.grid(True)
plt.show()

# converts data into numpy array
x = data_set.iloc[:, 0].values.reshape(-1, 1) #select all rows, columns year, appearances, align_b
y = data_set.iloc[:, 4].values.reshape(-1, 1)

# Split the data into training/testing sets
ds_sub_x_train = x[:-20]
ds_sub_x_test = x[-20:]

# Split the targets into training/testing sets
ds_sub_y_train = y[:-20]
ds_sub_y_test = y[-20:]

# create linear regression object
lin_regres = LinearRegression()

#Train the model using the training sets
lin_regres.fit(ds_sub_x_train, ds_sub_y_train)

#Make predictions using the testing set
y_pred = lin_regres.predict(ds_sub_x_test)
    

#Plot outputs
 