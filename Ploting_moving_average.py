import numpy as np
import pandas as pd

data = np.genfromtxt('Historical_eth-usd_data_hourly_step.txt',delimiter=',')

#CAPTURE THE A SPECIFIC FEATURE AND SET IT AS DATAFRAME
open_eth_data = pd.DataFrame({ 'Open' : data[1:,3]})
low_eth_data = pd.DataFrame({ 'Low' : data[1:,1]})
high_eth_data = pd.DataFrame({ 'High' : data[1:,2]})

#THE SIZE OF THE WINDOW FOR SMOOTHING
window=20

#TAKE THE PREVIOUS DATAFRAMES AND SMOOTH TO THE WINDOW SIZE
rm_eth_data = open_eth_data.rolling(window).mean()
rm_low_eth_data = low_eth_data.rolling(window).mean()
rm_high_eth_data = high_eth_data.rolling(window).mean()


#TAKE THE PREVIOUS ORIGINAL DATAFRAMES AND GET THE STD FOR THE WINDOW SIZE
std_open_eth_data = open_eth_data.rolling(window).std()
std_low_eth_data = low_eth_data.rolling(window).std()
std_high_eth_data = high_eth_data.rolling(window).std()

#THE AMOUNT OF STD FOR THE BOLLINGER BANDS
sigma = 2

#CREATING THE UPPER AND LOWER BOLLINGER BANDS (R)
upper_bollinger_bands_r = rm_eth_data + sigma*std_open_eth_data
lower_bollinger_bands_r = rm_eth_data - sigma*std_open_eth_data

#AMOUNT OF SIGNAL FROM STD TO BE ADDED TO SMOOTHED LOW OF HIGH
Amount = 1.5

#CREATING MY OWN INDICATOR 1
upper_bryant_band = rm_high_eth_data + 1.5*std_high_eth_data
lower_bryant_band = rm_low_eth_data - 1.5*std_low_eth_data

#CREATING MY OWN INDICATOR 2  
upper_eth = (open_eth_data - lower_bollinger_bands_r)
lower_eth = (upper_bollinger_bands_r - open_eth_data)

#CREATING A BINARY ARRAY OF VALUES ABOVE OR BELOW ZERO OF THE
#TWO PREVIOUS DATAFRAMES
upper_eth[upper_eth['Open']<=0] = -1
upper_eth[upper_eth['Open']>0] = 1

#THE FOLLOWNG LOOP WILL COUNT CONSECUTIVE BINARY VALUES AND ADD THEM UNTIL
#THE TREND CHANGES. 
interval=1
downs=1
n=6
Array = np.empty(shape=(int(upper_eth.shape[0]/n),n))
Array[:] = np.nan
j=0
k=0
#import time
for i in range(2,int(upper_eth.shape[0]/n)*n):
    #print (upper_eth['Open'][i]-upper_eth['Open'][i-1])
    #time.sleep(0.5)
    if (upper_eth['Open'][i]-upper_eth['Open'][i-1])==0:
        
        if upper_eth['Open'][i]==1:
            interval = interval + 1
            #print (interval)
        if upper_eth['Open'][i]==-1:
            downs = downs + 1
            #print (interval)
    elif (upper_eth['Open'][i]-upper_eth['Open'][i-1])==np.nan:
        g=0
    else:
        if upper_eth['Open'][i-1]==1:
            Array[j,k]=interval
            interval = 1
        elif upper_eth['Open'][i-1]==-1:
            Array[j,k] = downs
            downs = 1
        k=k+1
        if k==n:
            j=j+1
            k=0
Array2 = Array[~np.isnan(Array).any(axis=1)]

#DELETE THE LAST COLUMN BECAUSE I WILL LIKE TO PREDICT THE LENGTH OF THE
#NEXT PERIOD BEFORE THE VALUE FALLS BELOW SIGMA TIMES BELOW THE SMA
Array2 = np.delete(Array2,5,1)

#GET THE MAX VALUE FOR EACH COLUMN. SIMPLE NORMALIZATION
Max = Array2.max(axis=0)

#GET THE DIMENSION OF THE ARRAY TO SET AS MAX
Array2_size = Array2.shape[0]

#SET THE CUT POINT THE ARRAY ABOVE TO DIVIDE BETWEEN TRAIN AND TEST SETS
Break_point = 120

from sklearn import svm,preprocessing

#NORMALIZATION METHOD A
#std_scale = preprocessing.StandardScaler().fit(Array2[0:100,:])
#std_scale = preprocessing.StandardScaler().fit(Array2[0:100,:])
#X_train_std = std_scale.transform(Array2[0:100,:])
#X_test_std  = std_scale.transform(Array2[100:120,:])

#NORMALIZATION METHOD B
X_train_std = (Array2[0:Break_point,:]/Max)
X_test_std  = (Array2[Break_point:Array2_size,:]/Max)

#TRAINING A SUPPORT VECTOR MACHINE ON DATA GIVEN
clf = svm.SVR(C=3.05, cache_size=1000, coef0=1, degree=5, gamma=1.925, kernel='poly', epsilon=0.09, max_iter=-1, shrinking=True, tol=0.01, verbose=False)
clf.fit(X_train_std[0:100,0:4],X_train_std[0:100,4])

#PREDICT ON TEST DATA SET
results = clf.predict(X_test_std[:,0:4])
print ('Kernel:'+str(clf.kernel)+' C:'+str(clf.C)+' Degree:'+str(clf.degree)+' Gamma:'+str(clf.gamma)+' Coef:'+str(clf.coef0))
print (np.sqrt(np.square((results - X_test_std[:,4]).sum())/20))
print (results * Max[4])
print (X_test_std[:,4] * Max[4])


#CREATING ARRAYS OF ONLY POSITIVE OR NEGATIVE VALUES
tell_tail_sign_1_positive = open_eth_data - lower_bollinger_bands_r
tell_tail_sign_1_negative = open_eth_data - lower_bollinger_bands_r

tell_tail_sign_2_positive = upper_bollinger_bands_r - open_eth_data
tell_tail_sign_2_negative = upper_bollinger_bands_r - open_eth_data

tell_tail_sign_1_negative[tell_tail_sign_1_negative>0] = np.nan
tell_tail_sign_1_positive[tell_tail_sign_1_positive<=0] = np.nan

tell_tail_sign_2_negative[tell_tail_sign_2_negative>0] = np.nan
tell_tail_sign_2_positive[tell_tail_sign_2_positive<=0] = np.nan

#save some space
del data

#SET THE BOUNDARIES FOR PLOTTING
upper=20
lower=220

#PLOT 
import matplotlib.pyplot as plt
#plt.subplot(2,1,1)
plt.plot(open_eth_data['Open'][upper:lower], label='ETH data')
#plt.plot(rm_eth_data['Open'][upper:lower], label=str(window)+' step rolling mean data')
plt.plot(upper_bollinger_bands_r['Open'][upper:lower], label=str(window)+' step positive Bollinger Bands') 
plt.plot(lower_bollinger_bands_r['Open'][upper:lower], label=str(window)+' step negative Bollinger Bands') 
#plt.plot(high_eth_data['High'][upper:lower], label='High')
#plt.plot(low_eth_data['Low'][upper:lower], label='Low')
#plt.plot(rm_high_eth_data['High'][upper:lower], label=str(window)+' step high mean')
#plt.plot(rm_low_eth_data['Low'][upper:lower], label=str(window)+' step low mean')
#plt.plot(upper_bryant_band['High'][upper:lower], label=str(window)+' step Upper Bryant Band')
#plt.plot(lower_bryant_band['Low'][upper:lower], label=str(window)+' step Lower Bryant Band')
#plt.ylabel("Opening price [$]")
#plt.title('ETH')
#plt.legend(loc=0)

#plt.subplot(2,1,2)
#plt.plot(tell_tail_sign_1_positive['Open'][upper:lower], label='Lower')
#plt.plot(tell_tail_sign_1_negative['Open'][upper:lower], '.-',label='Lower', color='black')
#plt.plot(tell_tail_sign_2_positive['Open'][upper:lower], label='Lower')
#plt.plot(tell_tail_sign_2_negative['Open'][upper:lower], '.-',label='Lower', color='black')
#plt.plot(lower_eth['Open'][upper:lower])
#plt.plot(upper_eth['Open'][upper:lower])

#plt.xlabel("Time in hours")
#plt.ylabel("Difference [$]")

#plt.savefig('ETH',dpi=1000)
plt.show()