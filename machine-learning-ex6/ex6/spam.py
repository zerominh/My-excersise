import numpy as np  
import os
import pandas as pd  
import matplotlib.pyplot as plt  
# import seaborn as sb  
from scipy.io import loadmat  
from sklearn import svm

path_data_train = os.getcwd() + '\spamTrain.mat'
raw_data_train = loadmat(path_data_train)  

path_data_test = os.getcwd() + '\spamTest.mat'
raw_data_test = loadmat(path_data_test)

X_train = raw_data_train['X']
y_train = raw_data_train['y'].ravel()
X_test = raw_data_test['Xtest']
y_test = raw_data_test['ytest'].ravel()

svc = svm.SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)


correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_test)]
accuracy = sum(correct) /len(correct)
print(accuracy)