import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = os.getcwd() + '\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data = (data - data.mean()) / data.std() 
# print(data.head())
# print(data.describe())


def computeCost(X, theta, y):
	p = X*theta - y
	return 1/(2*len(X))*(p.T*p)


data.insert(0, 'Ones', 1)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)  
y = np.matrix(y.values)  
theta = np.matrix([[0.00],[0.00],[0.00]])

def gradientDescent(X, y, theta, alpha, iters):
	m = len(X)
	J = np.matrix(np.zeros((iters,1)))
	for i in range(iters):
		# theta_pre = theta
		# # print(theta_pre)
		# p = X.shape[1]
		# for j in range(p):
		# 	deriv = (X*theta_pre - y).T*(X[:, j])/m
		# 	theta[j,0] = theta_pre[j,0] - alpha*deriv
		theta = theta - alpha*X.T*(X*theta -y)/m
		J[i,0] = computeCost(X, theta, y)	
	return theta, J

alpha = 1
iters = 50
theta, cost = gradientDescent(X, y, theta, alpha, iters)
print(theta)
print(cost)
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 
ax.legend()
# plt.show()