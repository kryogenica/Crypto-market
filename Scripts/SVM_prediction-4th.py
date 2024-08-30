#Use Scikit learn to train various SVM
import numpy as np

#loads csv data into a ndarray
#np.random.seed(2018)

start=0
midrange=506
maxrange=606

my_data = np.genfromtxt('open_data_normed.csv',delimiter=',')[0:-1,:]
senti_data=np.genfromtxt('sentimental_arranged.txt',delimiter=',')
Means = np.genfromtxt('Means_open.csv',delimiter=',')[0:-1]
Stds = np.genfromtxt('Stds_open.csv',delimiter=',')[0:-1]

print (my_data.shape)
print (senti_data.shape)

X_train=my_data[start:midrange,0:-1]
senti_train=senti_data[start:midrange]
senti_train=np.reshape(senti_train,(senti_train.shape[0],1))
y_train=my_data[start:midrange,-1]

X_holdout=my_data[midrange:maxrange,0:-1]
senti_holdout=senti_data[midrange:maxrange]
senti_holdout=np.reshape(senti_holdout,(senti_holdout.shape[0],1))
y_holdout=my_data[midrange:maxrange,-1]

from sklearn import svm
#gamma=0.02 C=0.1
clf = svm.SVR(C=3.07, cache_size=1000, coef0=250, degree=3, gamma=30, kernel='rbf', epsilon=0.1, max_iter=-1, shrinking=True, tol=0.01, verbose=False)

A=np.hstack([X_train,senti_train])
#clf.fit(X_train,y_train) #Use this line instead of the one below if you
#do not wish to use sentimental data
clf.fit(A, y_train)

B=np.hstack([X_holdout,senti_holdout])
#y_pred=clf.predict(X_holdout) #Use this line instead of the one below if you
#do not wish to use sentimental data
y_pred=clf.predict(B)
    
Back_to_normal_pred=((np.multiply(y_pred,Stds[midrange:maxrange]))+Means[midrange:maxrange])
Back_to_normal_holdout=((np.multiply(y_holdout,Stds[midrange:maxrange]))+Means[midrange:maxrange])
    
    
import matplotlib.pyplot as plt
plt.plot(Back_to_normal_holdout[0:250], label='Real data')
plt.plot(Back_to_normal_pred[0:250], label='Predicted data')
plt.xlabel("Time in hours [21 minute intervals]")
plt.ylabel("Price at closing time [US dollars]")
plt.title('ETH - RBF kernel + Sentiment')
plt.legend(loc=1)
plt.savefig('ETH - RBF kernel - Sentiment',dpi=1000)
plt.show()

Dif=Back_to_normal_pred[0:100]-Back_to_normal_holdout[0:100]
x=np.linalg.norm(Dif)/100
print (x)

#print (np.linalg.norm(Dif)/100)
#print (sum/j)
#print (np.hstack(deviations).std())