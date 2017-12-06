import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import math
import os
import timeit


path = os.getcwd() + '\data3.txt'  
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
	# start = timeit.default_timer()
	cols = r.shape[1]
	n = int(persent*num_vectors)
	for i in range(num_vectors):
		eps =  np.sort(r[i,:], kind = 'heapsort')[n-1]
		for j in range(num_vectors):
			if(r[i,j] <= eps): r[i,j] = 1
			else: r[i,j] = 0;
	# stop = timeit.default_timer()
	# print(stop - start)
	return r

def recurrence(x, dim, tau, persent):
	# start = timeit.default_timer()
	n = x.shape[0]
	num_vectors = n - (dim-1)*tau
	r = np.zeros((num_vectors, num_vectors))
	for i in range (num_vectors):
		r[i,i] = 0
		for j in range(i+1, num_vectors):
			y = 0.00
			#distance between two vectors
			for k in range(dim):
				y += (x[i + k*tau] - x[j + k*tau])**2.00	
			r[i, j] = math.sqrt(y)
			r[j, i]	= r[i,j]
	# stop = timeit.default_timer()
	# print(stop - start)
	return create_recurrence_table(r, num_vectors, persent)
tau = 3
dim = 3
persent = 0.2
start = timeit.default_timer()
r = recurrence(x, dim, tau, persent)
stop = timeit.default_timer()
print(stop - start) 


#plot transpose of r
a = np.array(r.T)
rows = a.shape[0]
cols = a.shape[1]
iter_row_from_0= []
for i in range(int(rows/2)):
	iter_row_from_0.append(i)
iter_row_from_1 = []
for i in range(1,int(rows/2)+1):
	iter_row_from_1.append(-i)
for i, j in zip(iter_row_from_0, iter_row_from_1):
	temp = np.array(a[i,:])
	a[i,:] = a[j,:]
	a[j,:] = temp	
plt.imshow(a, interpolation='nearest', cmap=plt.cm.ocean,
    extent=(0.5,np.shape(a)[0]+0.5,0.5,np.shape(a)[1]+0.5))
# plt.matshow(r)
plt.show()




# print(r)
l = []
def calculate_l_table(recurrence_table):
	rows =  recurrence_table.shape[0]
	cols =  recurrence_table.shape[1]
	for b in range(-rows+1, 0):
		len = 0
		for x in range(-b, rows):
			if recurrence_table[x, x+b] == 1 : len  += 1;
			elif len != 0 and len != rows : l.append(len); len = 0;
		if len != 0 : l.append(len)
	for b in range(1, rows):
		len = 0
		for x in range(0, rows-b):
			if recurrence_table[x, x+b] == 1 : len += 1; 
			elif len != 0 : l.append(len); len = 0;
		if len != 0 : l.append(len)
	return l
# start = timeit.default_timer()
l_table = calculate_l_table(r)
# print(l_table)
# stop = timeit.default_timer()
# print(stop - start) 
frequence_l_table = [0]*(r.shape[0]+1)

for i in l_table:
	frequence_l_table[i] += 1;
# print(frequence_l_table)
def calculate_DET(l_table, N):
	return sum(l_table)/ float(N*N)


print('DET: ' +str(calculate_DET(l_table, r.shape[0])))


def calculate_MDL(l_table):
	return sum(l_table)/len(l_table)

print('MDL: ' + str(calculate_MDL(l_table)))

def calculate_ENTR(frequence_l_table, N):
	probability_l = [0]*(N+1)
	s = sum(frequence_l_table)
	for i in range(1, N+1):
		if(frequence_l_table[i] != 0): probability_l[i] = frequence_l_table[i]/s
	ENTR = 0.00
	for i in probability_l:
		if(i != 0): ENTR += i*np.log(i)
	return -ENTR

print('ENTR:  ' + str(calculate_ENTR(frequence_l_table, r.shape[0])))

#height

def calculate_height_table(recurrence_table):
	h = []
	rows = r.shape[0]
	len = 0
	for i in range(rows):
		for j in range(rows):
			if recurrence_table[j, i] == 1 : len  += 1;
			elif len != 0 : h.append(len); len = 0;
		if len != 0 : h.append(len); len = 0;
	return h

h = calculate_height_table(r)
# print(h)


def calculate_LAM(h_table):
	return sum(h_table)/(len(h_table)**2)

print("LAM: " + str(calculate_LAM(h)))

def calculate_TT(h_table):
	return sum(h_table)/(len(h_table))


print("TT: " + str(calculate_TT(h)))





