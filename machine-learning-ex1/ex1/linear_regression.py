# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = os.getcwd() + '\ex1data1.txt'  
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])  
data.head()  
data.describe()  
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8)) 
# plt.legend()
# plt.show()


def computeCost(X, theta, y):
	p = X*theta - y;
	return 1/(2*len(X))*(p.T*p);


data.insert(0, 'Ones', 1)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)  
y = np.matrix(y.values)  
theta = np.matrix(np.array([[0],[0]]))

J = computeCost(X, theta, y)
print(J)
