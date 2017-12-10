import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder




path = os.getcwd() + '\ex4data1.mat'
data = loadmat(path)

# print(data['X'].shape)
# print(data['y'].shape)
X = data['X']
y = data['y']


#convert y to 0 0 0 0 0 1
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y)

X = np.matrix(X)
y_one_hot = np.matrix(y_one_hot)


def sigmoid(z):
	return 1.00/(1.00 + np.exp(-z))


# def predict_all(X, all_theta):
# 	rows, cols = X.shape
# 	X = np.insert(X, 0, np.ones(rows), axis = 1)
# 	X = np.matrix(X)
# 	all_theta = np.matrix(all_theta)
# 	h = sigmoid(X*all_theta.T)
# 	h_argmax = np.argmax(h, axis = 1)
# 	return h_argmax + 1
#
#
# y_pred = predict_all(data['X'], all_theta)
# correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
# accuracy = sum(correct) /len(correct)
# print(accuracy)


def feed_forward(X, theta1, theta2):
    rows = X.shape[0]
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    a1 = X
    z2 = a1*theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a1, 0, values=1, axis=1)
