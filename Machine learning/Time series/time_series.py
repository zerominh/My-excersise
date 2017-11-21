import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = os.getcwd() + '\data.txt'  
data = pd.read_csv(path, header=None, names=['Time', 'Passenger'])
# print(data.head())
# print(data.describe())

# data.plot(x='Time', y='Passenger', figsize=(12,8)) 
# plt.show()

cols = data.shape[1]
x = data.iloc[:,cols-1:cols]
x = x.values
def f(x, m, to, eps):
	n = x.shape[0]
	result = np.array([])
	limit = n - (m-1)*to
	for i in range (limit):
		for j in range(i, limit):
			b = np.array([])
			for k in range(m):
				b = np.append(b, x[i + k*to] - x[j + k*to])
			if(eps > np.linalg.norm(b)):
				result = np.append(result, np.array([i,j]))
	return result
n = x.shape[0]
eps = 0.5*n
m = 3
to = 3
result = f(x, m, to, eps)
a =[]
b  = []
l = len(result)
for i in range(l):
	if(i % 2 == 0):
		a.append(int(result[i]))
	else:
		b.append(int(result[i]))

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(a, b, marker = '.')
plt.show()


