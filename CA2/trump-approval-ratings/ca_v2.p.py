import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

##SET DIRECTORY
os.getcwd()
os.chdir('C:/github/B9DA104/CA2/trump-approval-ratings/')
os.listdir()


################################################################

##read file
data = pd.read_csv('approval_polllist.csv')
type(data)


##converts data into numpy array
X = data.iloc[:, 12].values.reshape(-1, 1)
Y = data.iloc[:, 7].values.reshape(-1, 1)

##create variable for Lregression and execute
linRegres = LinearRegression()
linRegres.fit(X, Y)
yPred = linRegres.predict(X)  # make predictions

##print it


plt.title("Graph Disapprove") 
plt.xlabel("disapprove") 
plt.ylabel("population") 
#plt.scatter(X,Y)
plt.scatter(X, Y, color='black')
plt.plot(X, yPred, color='red', linewidth=3)
plt.show()


#### outro teste
# Load the diabetes dataset

data = pd.read_csv('approval_polllist.csv')
type(data)


##converts data into numpy array
X = data.iloc[:, 12].values.reshape(-1, 1)
Y = data.iloc[:, 7].values.reshape(-1, 1)

# Split the data into training/testing sets
data_x_train = X[:-20]
data_x_test = X[-20:]

# Split the targets into training/testing sets
data_y_train = Y[:-20]
data_y_test = Y[-20:]

#Create linear regression object
linRegres = LinearRegression()

#Train the model using the training sets
linRegres.fit(data_x_train, data_y_train)

#Make predictions using the testing set
yPred = linRegres.predict(data_x_test)

#Plot outputs
plt.scatter(data_x_test, data_y_test, color='black')
plt.plot(data_x_test, yPred, color='red', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.title("Graph Disapprove") 
plt.xlabel("disapprove") 
plt.ylabel("population") 

plt.show()



##print it
plt.scatter(X, Y)
plt.plot(X, yPred, color='red')
plt.show()

plt.title("Graph Disapprove") 
plt.xlabel("disapprove") 
plt.ylabel("population") 
plt.scatter(X,Y)