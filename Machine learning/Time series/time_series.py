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

def create_recurrence_table(r, num_vectors, persent):
	for i in range(num_vectors):
		n = int(np.absolute(persent*num_vectors-1))
		eps =  np.sort(r[i,:], kind='quicksort')[n]
		for j in range(num_vectors):
			if(r[i,j] <= eps): r[i,j] = 1
			else: r[i,j] = 0


def f(x, dim, tau, persent):
	n = x.shape[0]
	num_vectors = n - (dim-1)*tau
	r = np.zeros((num_vectors, num_vectors))
	for i in range (num_vectors):
		for j in range(num_vectors):
			y = np.array([])
			#distance between two vectors
			for k in range(dim):
				y = np.append(y, x[i + k*tau] - x[j + k*tau])
			r[i, j] = np.linalg.norm(y)		
	create_recurrence_table(r, num_vectors, persent)
	return r
tau = 3
dim = 3
persent = 0.2


r = f(x, dim, tau, persent)
num_vectors = r.shape[0]
a = np.array([])
b = np.array([])
for i in range(num_vectors):
	for j in range(num_vectors):
		if(r[i,j] == 1): a = np.append(a, i); b = np.append(b, j)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(a, b, marker = '.')
plt.show()



