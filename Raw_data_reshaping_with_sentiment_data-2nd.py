#Loads csv file into array only comformed by one feature
#The size of the rows are deterimined by the window_step size 
import numpy as np
import time

##########Difernece between High and low in time window

#loads csv data into a ndarray 
my_data = np.genfromtxt('Historical_eth-usd_data_minute_step.txt',delimiter=',')
sentiment = np.genfromtxt('clf_SVM_#ethereum_2018-04-11_2018-04-20.csv',delimiter=',')[1:,1:3]            
window_step=21 #Sets the window size

#print (sentiment[0,0])
#print (sentiment[-1,0])

#Set matrix A to contain rows of time frame Lows
feature=1 #Order: time,low,high,open,close,volume
prev_i=window_step+1 #Will help with setting the range when extracting data
#An array is initialized with the first data points
A=my_data[1:window_step+1,feature]

#range naturally ensures that the last vector is the same dim as the others
for i in range(((window_step)+(window_step)+1),my_data.shape[0],window_step):
    #Helps concatenate our vectors into an array
    A=np.r_['0,2',A,my_data[prev_i:i,feature]]
    prev_i=i #rest to previous i
Mins=(A[:,0:-1].min(axis=1).reshape(A.shape[0],1))

del A
#Set matrix A to contain rows of time frame Highs
feature=2 #Order: time,low,high,open,close,volume
prev_i=window_step+1 #Will help with setting the range when extracting data
#An array is initialized with the first data points
A=my_data[1:window_step+1,feature]

#range naturally ensures that the last vector is the same dim as the others
for i in range(((window_step)+(window_step)+1),my_data.shape[0],window_step):
    #Helps concatenate our vectors into an array
    A=np.r_['0,2',A,my_data[prev_i:i,feature]]
    prev_i=i #rest to previous i
#Get the maximum of each row
Maxs=(A[:,0:-1].max(axis=1).reshape(A.shape[0],1))

#Set a column to contain the dif between Highs and Lows.
#Normalized by minus mean and divided by std.
Reach=(Maxs-Mins)
Reach_info=np.array([Reach.mean(axis=0),Reach.std(axis=0)])
np.savetxt("reach_mean_and_std.csv", Reach_info, delimiter=",")
Reach=((Reach-(Reach).mean(axis=0))/(Reach.std(axis=0)))
del A,Maxs,Mins

################Reshaping dataset to contain data on opening prices and reach

feature=3 #Order: time,low,high,open,close,volume
prev_i=window_step+1 #Will help with setting the range when extracting data
#An array is initialized with the first data points
A=my_data[1:window_step+1,feature]
A[-1]=my_data[window_step+1,4]
j=1
senti=[]


#range naturally ensures that the last vector is the same dim as the others
for i in range(((window_step)+(window_step)+1),my_data.shape[0],window_step):
    #Helps concatenate our vectors into an array
    A=np.r_['0,2',A,my_data[prev_i:i,feature]]
    A[j,-1]=my_data[i,4]
    past=int(my_data[prev_i,0])
    future=int(my_data[i,0])
    suma=0
    l=0
    k=0
    b=0
    Z=[0]
    #print ("outside")    
    for k in range(0,sentiment.shape[0]):
        timestamp=int(sentiment[k,0]/1000)
        if (timestamp>=past):
            l=l+1
        if (timestamp<=future):
            if (l>0):
                if (timestamp>=past):
                    Z.append(sentiment[k,1])
        if (timestamp>=future):
            k=sentiment.shape[0]


    if (np.asanyarray(Z).shape[0]==1):
        senti.append(0)
    else :
        vr=np.asanyarray(Z)
        vr=np.delete(vr,0,0)
        senti.append(vr.mean())
    
    j=j+1
    prev_i=i #rest to previous i    
print (senti)

np.savetxt("sentimental.txt", senti, delimiter=",")
#We calculate the mean of each row without including the last value
Means=((A[:,0:-1].mean(axis=1)).reshape(A.shape[0],1))
np.savetxt("Means_open.csv", Means, delimiter=",")
Means=(np.repeat(Means,A.shape[1],axis=1))

Temp=(A[:,0:-1]-Means[:,0:-1])
Temp=np.square(Temp)

#We calculate the std of each row without including the last value
Stds=(Temp.std(axis=1).reshape(A.shape[0],1))
np.savetxt("Stds_open.csv", Stds, delimiter=",")
Stds=(np.repeat(Stds,A.shape[1],axis=1))

#All rows are normalized minus mean and divided by std.
Normed_stock=(np.divide((A-Means),Stds))
del A,Means,Stds,Temp

#Append vector and array. Order matters.
Full_stock=np.c_[Reach,Normed_stock]
np.savetxt("open_data_normed.csv", Full_stock, delimiter=",")
del Normed_stock
del Full_stock

#############Low Data

feature=1 #Order: time,low,high,open,close,volume
prev_i=window_step+1 #Will help with setting the range when extracting data
#An array is initialized with the first data points
A=my_data[1:window_step+1,feature]
#range naturally ensures that the last vector is the same dim as the others
for i in range(((window_step)+(window_step)+1),my_data.shape[0],window_step):
    #Helps concatenate our vectors into an array
    A=np.r_['0,2',A,my_data[prev_i:i,feature]]
    prev_i=i #rest to previous i

#We calculate the mean of each row without including the last value
Means=((A[:,0:-1].mean(axis=1)).reshape(A.shape[0],1))
np.savetxt("Means_low.csv", Means, delimiter=",")
Means=(np.repeat(Means,A.shape[1],axis=1))

Temp=(A[:,0:-1]-Means[:,0:-1])
Temp=np.square(Temp)

#We calculate the std of each row without including the last value
Stds=(Temp.std(axis=1).reshape(A.shape[0],1))
np.savetxt("Stds_low.csv", Stds, delimiter=",")
Stds=(np.repeat(Stds,A.shape[1],axis=1))

#All rows are normalized minus mean and divided by std.
Normed_stock=(np.divide((A-Means),Stds))

#Append vector and array. Order counts.
Full_stock=np.c_[Reach,Normed_stock]

np.savetxt("low_data_normed.csv", Full_stock, delimiter=",")
del Normed_stock
del Full_stock
#############High Data
del A
feature=2 #Order: time,low,high,open,close,volume
prev_i=window_step+1 #Will help with setting the range when extracting data
#An array is initialized with the first data points
A=my_data[1:window_step+1,feature]
#range naturally ensures that the last vector is the same dim as the others
for i in range(((window_step)+(window_step)+1),my_data.shape[0],window_step):
    #Helps concatenate our vectors into an array
    A=np.r_['0,2',A,my_data[prev_i:i,feature]]
    prev_i=i #rest to previous i

#We calculate the mean of each row without including the last value
Means=((A[:,0:-1].mean(axis=1)).reshape(A.shape[0],1))
np.savetxt("Means_high.csv", Means, delimiter=",")
Means=(np.repeat(Means,A.shape[1],axis=1))

Temp=(A[:,0:-1]-Means[:,0:-1])
Temp=np.square(Temp)

#We calculate the std of each row without including the last value
Stds=(Temp.std(axis=1).reshape(A.shape[0],1))
np.savetxt("Stds_high.csv", Stds, delimiter=",")
Stds=(np.repeat(Stds,A.shape[1],axis=1))

#All rows are normalized minus mean and divided by std.
Normed_stock=(np.divide((A-Means),Stds))

#Append vector and array. Order counts.
Full_stock=np.c_[Reach,Normed_stock]

np.savetxt("high_data_normed.csv", Full_stock, delimiter=",")