import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import math
import os
import timeit


# path = os.getcwd() + '\data2.txt'  
# data = pd.read_csv(path, header=None)
# # print(data.head())
# # print(data.describe())
# rows = data.shape[0]
# cols = data.shape[1]
# x = data.iloc[:,cols-1:cols]
# x = x.values
# fig, ax = plt.subplots(figsize=(12,12))
# ax.plot(np.arange(x.shape[0]), x, 'r') 
# plt.show()



#generate data
volume = 0.5     # range [0.0, 1.0]
fs = 44100      # sampling rate, Hz, must be integer
duration = 0.1   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples = (np.tan(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(np.arange(len(samples)), samples, 'r') 
plt.show()

def create_recurrence_table(r, num_vectors, persent):
	# start = timeit.default_timer()
	cols = r.shape[1]
	n = int(persent*num_vectors)
	if(n == 0): n = 1;
	# print(n)
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
	# print("  ban dau: " + str(r))
	return create_recurrence_table(r, num_vectors, persent)
tau = 90
dim = 40
persent = 0.01
start = timeit.default_timer()
r = recurrence(samples, dim, tau, persent)
stop = timeit.default_timer()
print(stop - start) 


#plot transpose of r
# a = np.array(r.T)
# rows = a.shape[0]
# cols = a.shape[1]
# iter_row_from_0= []
# for i in range(int(rows/2)):
# 	iter_row_from_0.append(i)
# iter_row_from_1 = []
# for i in range(1,int(rows/2)+1):
# 	iter_row_from_1.append(-i)
# for i, j in zip(iter_row_from_0, iter_row_from_1):
# 	temp = np.array(a[i,:])
# 	a[i,:] = a[j,:]
# 	a[j,:] = temp	
# plt.imshow(a, interpolation='nearest', cmap=plt.cm.ocean,
#     extent=(0.5,np.shape(a)[0]+0.5,0.5,np.shape(a)[1]+0.5))
# # plt.matshow(r)
# plt.show()
a = []
b = []
n = r.shape[0]
for i in range(n):
	for j in range(n):
		if(r[i, j] == 1):
			a.append(i)
			b.append(j)

plt.scatter(a, b, s = 0.1, marker = '.')
plt.show()

# print(r)
def calculate_l_table(recurrence_table):
	frequence_l_table = [0]*(recurrence_table.shape[0])
	rows =  recurrence_table.shape[0]
	cols =  recurrence_table.shape[1]
	for b in range(-rows+1, 0):
		len = 0
		for x in range(-b, rows):
			if recurrence_table[x, x+b] == 1 : len  += 1;
			elif len != 0: frequence_l_table[len] += 1;len = 0; 
		if len != 0 : frequence_l_table[len] += 1
	for b in range(1, rows):
		len = 0
		for x in range(0, rows-b):
			if recurrence_table[x, x+b] == 1 : len += 1; 
			elif len != 0 : frequence_l_table[len] += 1;len = 0;
		if len != 0 : frequence_l_table[len] += 1
	return frequence_l_table
# start = timeit.default_timer()
frequence_l_table = calculate_l_table(r)
# print(l_table)
# stop = timeit.default_timer()
# print(stop - start) 

# print(frequence_l_table)
def calculate_DET(frequence_l_table):
	s = 0.00
	n = len(frequence_l_table)
	for i in range(n):
		if(frequence_l_table[i] != 0): s += frequence_l_table[i]*i;
	return s/float(n*n)


print('DET: ' +str(calculate_DET(frequence_l_table)))

def calculate_L(frequence_l_table):
	n = len(frequence_l_table)
	for i in range(n-1, 0, -1):
		if(frequence_l_table[i] > 0):
			return i
	return 0

print('L: ' +str(calculate_L(frequence_l_table)))
def calculate_MDL(frequence_l_table):
	s = 0.00
	num_l = 0
	n = len(frequence_l_table)
	for i in range(n):
		if(frequence_l_table[i] != 0): 
			s += frequence_l_table[i]*i
			num_l += frequence_l_table[i]
	return s/num_l

print('MDL: ' + str(calculate_MDL(frequence_l_table)))

def calculate_ENTR(frequence_l_table):
	N = len(frequence_l_table)
	probability_l = [0]*(N)
	s = sum(frequence_l_table)
	for i in range(1, N):
		if(frequence_l_table[i] != 0): probability_l[i] = frequence_l_table[i]/s
	ENTR = 0.00
	for i in probability_l:
		if(i != 0): ENTR += i*np.log(i)
	return -ENTR

print('ENTR:  ' + str(calculate_ENTR(frequence_l_table)))

#height

def calculate_height_table(recurrence_table):
	h = []
	rows = r.shape[0]
	len = 0
	for i in range(rows):
		len = 0
		for j in range(rows):
			if recurrence_table[i, j] == 1 : len  += 1;
			elif len != 0 : h.append(len); len = 0;
		if len != 0 : h.append(len);
	return h

h = calculate_height_table(r)
# print(h)
print("V: " + str(np.max(h)))

def calculate_LAM(h_table, n):
	return float(sum(h_table))/(n*n)

print("LAM: " + str(calculate_LAM(h, r.shape[0])))

def calculate_TT(h_table):
	return sum(h_table)/(len(h_table))


print("TT: " + str(calculate_TT(h)))


# def calculate_T1(recurrence_table):
# 	T1 = []
# 	n = len(recurrence_table)
# 	sum_t = 0
# 	prev_t = 0
# 	curr_t = 0
# 	for i in range(n):
# 		for j in range(n):
			
			

