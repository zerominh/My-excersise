import numpy as np  
import os
import pandas as pd  
import matplotlib.pyplot as plt  
# import seaborn as sb  
from scipy.io import loadmat  
from sklearn import svm

# path = os.getcwd() + '\ex6data1.mat'
# raw_data = loadmat(path)  

# data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  
# data['y'] = raw_data['y']




# positive = data[data['y'].isin([1])]  
# negative = data[data['y'].isin([0])]
 
# plt.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')  
# plt.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')  




# svc = svm.LinearSVC(C = 1.0)
# svc.fit(data[['X1', 'X2']], data['y'])
# # print(svc.score(data[['X1', 'X2']], data['y']))

# data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
# plt.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
# # plt.show()

# svc2 = svm.LinearSVC(C = 100)
# svc2.fit(data[['X1', 'X2']], data['y'])
# data['SVM 2 Confidence'] = svc.decision_function(data[['X1', 'X2']])
# plt.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
# # plt.show()


# path = os.getcwd() + '\ex6data2.mat'
# raw_data2 = loadmat(path)

# data2 = pd.DataFrame(raw_data2['X'], columns=['X1', 'X2'])  
# data2['y'] = raw_data2['y']

# positive = data2[data2['y'].isin([1])]  
# negative = data2[data2['y'].isin([0])]

# plt.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')  
# plt.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')  
# # plt.show()


# svc3 = svm.SVC(C=100, gamma=10, probability=True)  
# svc3.fit(data2[['X1', 'X2']], data2['y'])  
# data2['Probability'] = svc3.predict_proba(data2[['X1', 'X2']])[:,0]  
# plt.scatter(data2['X1'], data2['X2'], s=50, c=data2['Probability'], cmap='Reds')
# plt.show()


path = os.getcwd() + '\ex6data3.mat'
raw_data3 = loadmat(path)
X = raw_data3['X']
y = raw_data3['y'].ravel()
Xval = raw_data3['Xval']
yval = raw_data3['yval'].ravel()
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  
gamma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

def choose_param(X_train, y_train, X_cv, y_cv, C, gamma):
	best_score = 0.00
	best_params = {'C' : None, 'gamma' : None}
	for c in C:
		for g in gamma:
			svc = svm.SVC(C = c, gamma = g)
			svc.fit(X_train, y_train)
			score = svc.score(X_cv, y_cv)
			if(best_score < score):
				best_score = score
				best_params['C'] = c
				best_params['gamma'] = g
	return best_score, best_params


best_score, best_params = choose_param(X, y, Xval, yval, C, gamma)

svc = svm.SVC(C = best_params['C'], gamma = best_params['gamma'], probability = True)
svc.fit(X, y)
prob = svc.predict_proba(X)[:, 0]
positive = np.zeros((X.shape))
negative = np.zeros((X.shape))
for i in range(y.shape[0]):
	if(y[i] == 1):
		positive[i, :] = X[i, :]
	else: negative[i, :] = X[i, :]

# plt.scatter(positive[:,0], positive[:,1], s=30, marker='x', label='Positive')  
# plt.scatter(negative[:,0], negative[:,1], s=30, marker='o', label='Negative')  
plt.scatter(X[:, 0], X[:, 1], s=50, c = prob, cmap='Reds')
plt.show()

