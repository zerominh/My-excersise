import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = os.getcwd() + '\data.txt'  
data = pd.read_csv(path, header=None, names=['Time', 'Passenger'])
# print(data.head())
# print(data.describe())

data.plot(x='Time', y='Passenger', figsize=(12,8)) 
# plt.show()

cols = data.shape[1]
a = data.iloc[:,cols-1:cols]
a = a.values

print(a[0,0])