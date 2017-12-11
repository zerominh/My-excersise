import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from scipy.optimize import minimize
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder




path = os.getcwd() + '\ex4data1.mat'
data = loadmat(path)

# print(data['X'].shape)
# print(data['y'].shape)
X = data['X']
y = data['y']



path1 = os.getcwd() + '\ex4weights.mat'
data1 = loadmat(path1)
Theta1 = np.matrix(data1['Theta1'])
Theta2 = np.matrix(data1['Theta2'])

#convert y to 0 0 0 0 0 1
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y)

X = np.matrix(X)
y_one_hot = np.matrix(y_one_hot)


def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sigmoid_gradient(z):
	return np.multiply(sigmoid(z), 1- sigmoid(z))



def back_propagation(params, input_size, hidden_size, num_labels, X, y_training, learning_rate = 0):

	theta1 = np.matrix(np.reshape(params[:(hidden_size*(input_size+1))], (hidden_size, input_size+1)))
	theta2 = np.matrix(np.reshape(params[hidden_size*(input_size+1):], (num_labels, hidden_size+1)))
	tri_delta1 = np.zeros(theta1.shape)#25x401
	tri_delta2 = np.zeros(theta2.shape)#10x26


	rows = X.shape[0]
	m = X.shape[0]
	X = np.insert(X, 0, values=np.ones(rows), axis=1)#5000x401
	a1 = X
	z2 = a1*theta1.T#5000x25
	a2 = sigmoid(z2)
	a2 = np.insert(a2, 0, values=1, axis=1)#5000x26
	z2 = np.insert(z2, 0, values=1, axis=1)
	z3 = a2*theta2.T
	h = sigmoid(z3)
	first_term = np.multiply(-y_training, np.log(h))
	second_term = np.multiply(1-y_training, np.log(1-h))
	J = np.sum(first_term - second_term)/m
	reg = float(learning_rate)*(np.sum(np.power(theta1[:, 1:theta1.shape[1]], 2)) + np.sum(np.power(theta2[:, 1:theta2.shape[1]], 2)))/(2*m)
	J = J + reg
	#5000x10
	delta3 = h - y_training
	#5000x26
	delta2 = np.multiply(delta3*theta2, sigmoid_gradient(z2))

	tri_delta1 = delta2[:, 1:].T*a1
	tri_delta2 = delta3.T*a2

	D1 = tri_delta1/m
	D2 = tri_delta2/m

	D1[:, 1:] = D1[:, 1:] + learning_rate*(theta1[:, 1:])/m
	D2[:, 1:] = D2[:, 1:] + learning_rate*(theta2[:, 1:])/m


	grad = np.concatenate((np.ravel(D1), np.ravel(D2)))

	return J, grad



input_size = 400  
hidden_size = 25  
num_labels = 10  
learning_rate = 1

# randomly initialize a parameter array of the size of the full network's parameters
epsilon_init = math.sqrt(6)/(math.sqrt(input_size)+ math.sqrt(hidden_size))
theta1 = np.random.random(size=hidden_size * (input_size + 1))*2*epsilon_init - epsilon_init
epsilon_init = math.sqrt(6)/(math.sqrt(hidden_size)+ math.sqrt(num_labels))
theta2 = np.random.random(size=num_labels * (hidden_size + 1))*2*epsilon_init - epsilon_init
params = np.concatenate((theta1,theta2))

# cost, grad = back_propagation(params, input_size, hidden_size, num_labels, X, y_one_hot, learning_rate)
# print(grad)

fmin = minimize(fun=back_propagation, x0=params, args=(input_size, hidden_size, num_labels, X, y_one_hot, learning_rate), 
	method='TNC', jac=True, options={'maxiter': 100})
print(fmin)


def predict_all(X, theta1, theta2):
	rows, cols = X.shape
	X = np.insert(X, 0, np.ones(rows), axis = 1)
	X = np.matrix(X)
	h1 = sigmoid(X*theta1.T)
	h1 = np.insert(h1, 0, values = 1, axis = 1)
	h2 = sigmoid(h1*theta2.T)
	h_argmax = np.argmax(h2, axis = 1)
	return h_argmax + 1

theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
y_pred = predict_all(data['X'], theta1, theta2)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = sum(correct) /len(correct)
print(accuracy)