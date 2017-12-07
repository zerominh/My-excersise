import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.io import loadmat 

path = os.getcwd() + '\ex3data1.mat'
data = loadmat(path)

# print(data['X'].shape)
# print(data['y'].shape)
X = np.matrix(data['X'])
y = np.matrix(data['y'])

def sigmoid(z):
	return 1.00/(1.00 + np.exp(-z))


def lrCostFunction(theta, X, y, lamb):
	theta = np.matrix(theta)
	h = sigmoid(X*theta.T)
	m = len(X)
	reg = (lamb / (2*m)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
	return np.sum(np.multiply(-y,np.log(h)) - np.multiply(1-y,np.log(1-h)))/m + reg

def gradient_with_loop(theta, X, y, learningRate):
	theta = np.matrix(theta)
	parameters = int(theta.ravel().shape[1])
	grad = np.zeros(parameters)
	error = sigmoid(X * theta.T) - y
	for i in range(parameters):
		term = np.multiply(error, X[:,i])
		if (i == 0):
			grad[i] = np.sum(term) / len(X)
		else:
			grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
	return grad

def gradient(theta, X, y, learningRate):
	theta = np.matrix(theta)
	error = sigmoid(X*theta.T) - y
	grad = ((X.T*error).T + learningRate*theta)/len(X)
	grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
	return np.array(grad).ravel()
# theta_t = np.array([-2, -1, 1, 2])
# X_t = np.matrix(np.array([[1, 0.1, 0.14, 0.13],[1, 0.12, 0.13, 0.5], [1, 0.15, 0.15, 0.5], [1, 0.1, 0.11, 0.14], [1, 0.1, 0.1, 0.12]]))
# y_t = np.matrix(np.array([[1],[0],[1],[0],[1]]))
# lambda_t = 3


def one_vs_all(X, y, num_labels, learning_rate):
	rows = X.shape[0]
	params = X.shape[1]
	# k X (n + 1) array for the parameters of each of the k classifiers
	all_theta = np.zeros((num_labels, params + 1))
	# insert a column of ones at the beginning for the intercept term
	X = np.insert(X, 0, values=np.ones(rows), axis=1)
	# labels are 1-indexed instead of 0-indexed
	for i in range(1, num_labels + 1):
		theta = np.zeros(params + 1)
		y_i = np.array([1 if label == i else 0 for label in y])
		y_i = np.reshape(y_i, (rows, 1))
		# minimize the objective function
		fmin = minimize(fun=lrCostFunction, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient_with_loop)
		all_theta[i-1,:] = fmin.x
	return all_theta

all_theta = one_vs_all(np.matrix(data['X']), np.matrix(data['y']), 10, 1)

def predict_all(X, all_theta):
	rows, cols = X.shape
	X = np.insert(X, 0, np.ones(rows), axis = 1)
	X = np.matrix(X)
	all_theta = np.matrix(all_theta)
	h = sigmoid(X*all_theta.T)
	h_argmax = np.argmax(h, axis = 1)
	return h_argmax + 1


y_pred = predict_all(data['X'], all_theta)  
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]  
accuracy = sum(correct) /len(correct)  
print(accuracy)


