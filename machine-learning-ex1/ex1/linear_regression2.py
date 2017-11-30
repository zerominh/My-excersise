import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = os.getcwd() + '\ex1data1.txt'  
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# data.insert(0, 'Ones', 1)


cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)  
y = np.matrix(y.values)  


model = linear_model.LinearRegression()  
model.fit(X, y)  
f = model.predict(X).flatten()
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.plot(X[:, 1], f, 'r', label='Prediction')
# ax.legend() 
# plt.show()


print(model.coef_)
print(mean_squared_error(f, y))
