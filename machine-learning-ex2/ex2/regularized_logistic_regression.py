# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.optimize as opt

path = os.getcwd() + '\ex2data2.txt'  
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data[data['Accepted'].isin([1])]  
negative = data[data['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(positive['Test 1'], positive['Test 2'], s = 50, c='b', marker = 'x', label = 'Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s = 50, c='r', marker = 'o', label = 'Not') 
# plt.show()

data.insert(3, 'Ones', 1)
# print(data.head())
  
x1 = data['Test 1']
x2 = data['Test 2']  

degree = 6

for i in range(1, degree+1):
	for j in range(0, i+1):
		data['F' + str(i-j) +str(j)] = np.power(x1, i-j)+np.power(x2, j)


data.drop('Test 1', axis=1, inplace=True)  
data.drop('Test 2', axis=1, inplace=True)
print(data.head())
cols = data.shape[1]  
X = data.iloc[:,1:cols]  
y = data.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)  
y = np.array(y.values)
X = np.matrix(X)
y = np.matrix(y)

def sigmoid(z):
	return 1.00/(1.00 + np.exp(-z))
	
def costFunctionReg(theta, X, y, lamb):
	theta = np.matrix(theta)
	h = sigmoid(X*theta.T)
	m = len(X)
	reg = (lamb / (2*m)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
	return np.sum(np.multiply(-y,np.log(h)) - np.multiply(1-y,np.log(1-h)))/m + reg

# def gradientReg(theta, X, y, lamb):
# 	theta = np.matrix(theta)
# 	parameters = int(theta.shape[1])
# 	grad = np.zeros(parameters)
# 	error = sigmoid(X * theta.T) - y
# 	for i in range(parameters):
# 		term = np.multiply(error, X[:,i])
# 		if(i == 0): grad[i] = np.sum(term) / len(X);
# 		else: grad[i] = (np.sum(term) / len(X)) + (lamb*theta[:,i])/len(X);
# 	return grad

def gradientReg(theta, X, y, learningRate):  
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

lamb = 1
initial_theta = np.zeros((1,X.shape[1]))
# print(gradientReg(initial_theta, X, y, lamb))
result = opt.fmin_tnc(func = costFunctionReg, x0=initial_theta, fprime=gradientReg, args=(X, y, lamb))
print(result)

def predict(theta, X):
	probability = sigmoid(X*theta.T)
	return [1 if i >= 0.5 else 0 for i in probability]

theta_min = np.matrix(result[0])

predictions = predict(theta_min, X)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct)/len(correct)
print(accuracy)

# fig, ax = plt.subplots(figsize=(12,12))
# ax.scatter(positive['Test 1'], positive['Test 2'], s = 50, c='b', marker = 'x', label = 'Accepted')
# ax.scatter(negative['Test 1'], negative['Test 2'], s = 50, c='r', marker = 'o', label = 'Not') 
# ax.plot(X, predictions)
# plt.show()