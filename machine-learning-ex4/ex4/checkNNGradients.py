import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from scipy.optimize import minimize
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

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
	first_term = np.multiply(-y_training,np.log(h))
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


def debug_initialize_weights(fan_in, fan_out):
	W = np.zeros((fan_out, fan_in+1))
	W = np.reshape(np.sin(np.arange(fan_out*(fan_in+1))), (fan_out, fan_in+1))
	return W/10
	
def computeNumericalGradient(theta, input_layer_size, hidden_layer_size, number_labels, X, y, learning_rate):
	numgrad = np.zeros(theta.shape)
	perturb = np.zeros(theta.shape)
	e = 0.0001
	for i in range(theta.shape[0]):
		perturb[i] = e
		loss1, grad1 = back_propagation(theta -perturb, input_layer_size, 
			hidden_layer_size, number_labels, X, y, learning_rate)
		loss2, grad2 = back_propagation(theta +perturb, input_layer_size, 
			hidden_layer_size, number_labels, X, y, learning_rate)
		numgrad[i] = (loss2 - loss1)/(2*e)
		perturb[i] = 0
	return numgrad
def check_nn_gradients():
	input_layer_size = 3
	hidden_layer_size = 5
	number_labels = 3
	m = 5
	Theta1 = debug_initialize_weights(input_layer_size, hidden_layer_size)
	Theta2 = debug_initialize_weights(hidden_layer_size, number_labels)
	X = np.matrix(debug_initialize_weights(input_layer_size -1, m))
	print(X.shape)
	y = np.zeros((m, 1))
	for i in range(m):
		y[i, 0] = i % number_labels
	encoder = OneHotEncoder(sparse=False)
	y = encoder.fit_transform(y)
	print(y.shape)
	numgrad = computeNumericalGradient(np.concatenate((np.ravel(Theta1), np.ravel(Theta2))),
		input_layer_size, hidden_layer_size, number_labels, X, y, 0)
	cost, grad = back_propagation(np.concatenate((np.ravel(Theta1), np.ravel(Theta2))),
		input_layer_size, hidden_layer_size, number_labels, X, y, 0)
	return(np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad+grad))

print(check_nn_gradients())