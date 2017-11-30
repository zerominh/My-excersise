import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import math
import os
import timeit


path = os.getcwd() + '\data.txt'  
data = pd.read_csv(path, header=None)
# print(data.head())
# print(data.describe())
rows = data.shape[0]
cols = data.shape[1]
x = data.iloc[:,cols-1:cols]
x = x.values
# fig, ax = plt.subplots(figsize=(12,12))
# ax.plot(np.arange(x.shape[0]), x, 'r') 
# plt.show()


def create_recurrence_table(r, num_vectors, persent):
	recurrence_table =  np.zeros((num_vectors, num_vectors))
	for i in range(num_vectors):
		n = int(persent*num_vectors)
		idex_arr = np.argsort(r[i,:])
		for j in range(n):
			recurrence_table[i,idex_arr[j]] = 1;
	return recurrence_table

def recurrence(x, dim, tau, persent):
	n = x.shape[0]
	num_vectors = n - (dim-1)*tau
	r = np.zero((num_vectors, num_vectors))
	for i in range (num_vectors):
		r[i,i] = 0
		for j in range(i+1, num_vectors):
			y = 0.00
			#distance between two vectors
			for k in range(dim):
				y += (x[i + k*tau] - x[j + k*tau])**2.00	
			r[i, j] = math.sqrt(y)
			r[j, i]	= r[i,j]
	return create_recurrence_table(r, num_vectors, persent)
tau = 2
dim = 3
persent = 0.1

start = timeit.default_timer()
r= recurrence(x, dim, tau, persent)
stop = timeit.default_timer()
print(stop - start) 
# plt.imshow(r, interpolation='nearest', cmap=plt.cm.ocean,
    # extent=(0.5,np.shape(r)[0]+0.5,0.5,np.shape(r)[1]+0.5))
plt.matshow(r)
plt.show()
# print(r)
