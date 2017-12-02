import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.optimize as opt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = os.getcwd() + '\ex2data2.txt'  
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

# positive = data[data['Accepted'].isin([1])]  
# negative = data[data['Accepted'].isin([0])]
# fig, ax = plt.subplots(figsize=(12,12))
# ax.scatter(positive['Test 1'], positive['Test 2'], s = 50, c='b', marker = 'x', label = 'Accepted')
# ax.scatter(negative['Test 1'], negative['Test 2'], s = 50, c='r', marker = 'o', label = 'Not') 
# plt.show()

# print(data.head())

# convert to numpy arrays and initalize the parameter array theta
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]
print(y)
model = linear_model.LogisticRegression()  
model.fit(X, y)
f = model.predict(X)
def predict(f):
	return [1 if i >= 0.5 else 0 for i in f]

predictions = predict(f)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct)/len(correct)
print(accuracy)
