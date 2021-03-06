# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.optimize as opt

path = os.getcwd() + '\ex2data1.txt'  
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


# positive = data[data['Admitted'].isin([1])]  
# negative = data[data['Admitted'].isin([0])]
# fig, ax = plt.subplots(figsize=(12,12))
# ax.scatter(positive['Ex1'], positive['Ex2'], marker = 'x', label = 'Admitted')
# ax.scatter(negative['Ex1'], negative['Ex2'], marker = 'o', label = 'Not') 
# plt.show()

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)  
y = np.array(y.values)
X = np.matrix(X)
y = np.matrix(y)
def sigmoid(z):
	return 1.00/(1.00 + np.exp(-z))

def costFunction(theta, X, y):
	theta = np.matrix(theta)
	h = sigmoid(X*theta.T)
	m = len(X)
	return ((-y).T*np.log(h) - (1-y).T*np.log(1-h))/m

def gradient(theta, X, y):

	theta = np.matrix(theta)
	parameters = int(theta.shape[1])
	grad = np.zeros(parameters)
	error = sigmoid(X * theta.T) - y
	for i in range(parameters):
		term = np.multiply(error, X[:,i])
		grad[i] = np.sum(term) / len(X)
	return grad

initial_theta = np.zeros((1,X.shape[1]))
# print(gradient(initial_theta, X, y))
result = opt.fmin_tnc(func=costFunction, x0=initial_theta, fprime=gradient, args=(X, y))

def predict(theta, X):
	probability = sigmoid(X*theta.T)
	return [1 if i >= 0.5 else 0 for i in probability]

theta_min = np.matrix(result[0])

predictions = predict(theta_min, X)
# a and b equals (a == 1 and b == 1) or (a == 0 and b == 0)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct)/len(correct)
print(accuracy)



