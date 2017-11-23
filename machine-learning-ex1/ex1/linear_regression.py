# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = os.getcwd() + '\ex1data1.txt'  
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])  
# print(data.head())  
# print(data.describe())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8)) 
plt.legend()
plt.show()

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
# print(X)
# J = computeCost(X, theta, y)
# print(J)
def gradientDescent(X, y, theta, alpha, iters):
	m = len(X)
	J = np.matrix(np.zeros((iters,1)))
	for i in range(iters):
		theta = theta - alpha*(X.T *(X*theta - y))/m
		J[i,0] = computeCost(X,theta, y)
	return theta, J


alpha = 0.01
iters = 1000
theta, cost = gradientDescent(X, y, theta, alpha, iters)
# print(cost)
print(theta)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.plot(X[:, 1], X*theta, 'r', label='Prediction')
# ax.legend() 
# #plt.show()

# fig, ax = plt.subplots(figsize=(12,8))  
# ax.plot(np.arange(iters), cost, 'r')  
# ax.set_xlabel('Iterations')  
# ax.set_ylabel('Cost')  
# ax.set_title('Error vs. Training Epoch') 
# ax.legend()
# #plt.show()
