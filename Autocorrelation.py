import numpy as np
import pandas as pd

data = np.genfromtxt('Historical_eth-usd_data_hourly_step.txt',delimiter=',')

#CAPTURE THE A SPECIFIC FEATURE AND SET IT AS DATAFRAME
open_eth_data = pd.DataFrame({ 'Open' : data[1:,3]})
low_eth_data = pd.DataFrame({ 'Low' : data[1:,1]})
high_eth_data = pd.DataFrame({ 'High' : data[1:,2]})
close_eth_data = pd.DataFrame({ 'Close' : data[1:,2]})
rang_eth_data = high_eth_data['High'] - low_eth_data['Low']


#HEIKIN-ASHI DATA
open_heikin = (open_eth_data['Open'][0:-2] + close_eth_data['Close'][0:-2])/2
close_heikin = (open_eth_data['Open'][1:-1]+close_eth_data['Close'][1:-1]+low_eth_data['Low'][1:-1]+high_eth_data['High'][1:-1])/4
high_heikin = np.maximum.reduce([open_eth_data['Open'][1:-1],close_eth_data['Close'][1:-1],high_eth_data['High'][1:-1]])
low_heikin = np.minimum.reduce([low_eth_data['Low'][1:-1],close_eth_data['Close'][1:-1],open_eth_data['Open'][1:-1]])




#THE SIZE OF THE WINDOW FOR SMOOTHING
window=20

#TAKE THE PREVIOUS DATAFRAMES AND SMOOTH TO THE WINDOW SIZE
rm_eth_data = open_eth_data.rolling(window).mean()
rm_low_eth_data = low_eth_data.rolling(window).mean()
rm_high_eth_data = high_eth_data.rolling(window).mean()
rm_rang_eth_data = rang_eth_data.rolling(window).mean()

#TAKE THE PREVIOUS DATAFRAMES AND SMOOTH TO THE WINDOW SIZE
heikin_rm_eth_data = open_heikin.rolling(window).mean()

#TAKE THE PREVIOUS ORIGINAL DATAFRAMES AND GET THE STD FOR THE WINDOW SIZE
std_open_eth_data = open_eth_data.rolling(window).std()
std_low_eth_data = low_eth_data.rolling(window).std()
std_high_eth_data = high_eth_data.rolling(window).std()

#THE AMOUNT OF STD FOR THE BOLLINGER BANDS
sigma = 2

#CREATING THE UPPER AND LOWER BOLLINGER BANDS (R)
upper_bollinger_bands_r = rm_eth_data + sigma*std_open_eth_data
lower_bollinger_bands_r = rm_eth_data - sigma*std_open_eth_data

upper = 3850
lower = 3800

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fun(x,a,b,c,d):
    return ((d*((np.exp(-1*b*x))*(np.cos(c*x))))+a)

popt, pcov = curve_fit(fun,range(0,upper-lower),rm_eth_data['Open'][lower-1:upper-1],bounds=([3.,0.,0.,0.],[15.,2.,2.,1.]))
#print (popt)

sigmoids = np.array([])
for i in range(window,rang_eth_data.shape[0]):
    if (1/(1+ np.exp(-rang_eth_data[i] + 4*rm_rang_eth_data[i])))>0.45:
        sigmoids = np.append(sigmoids,i)
print (sigmoids.shape[0])

Errors = np.array([])
for i in range(1,11):
    
    z = np.polyfit(range(0,upper-lower),open_eth_data['Open'][lower-1:upper-1],i)
    p = np.poly1d(z)
    square_error = np.abs(open_eth_data['Open'][lower-1:upper-1] - p(range(0,upper-lower))).sum()
    Errors = np.append(Errors,square_error)







z = np.polyfit(range(0,upper-lower),open_eth_data['Open'][lower-1:upper-1],10)
p = np.poly1d(z)

#PLOT 
#import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc
fig1, ax = plt.subplots()
#candlestick2_ohlc(ax,open_eth_data['Open'][lower:upper],high_eth_data['High'][lower:upper],low_eth_data['Low'][lower:upper],close_eth_data['Close'][lower:upper],width=0.6)
#ax.plot(range(0,upper-lower),open_eth_data['Open'][lower-1:upper-1])
#ax.plot(range(0,upper-lower),lower_bollinger_bands_r['Open'][lower-1:upper-1])
#ax.plot(range(0,upper-lower),upper_bollinger_bands_r['Open'][lower-1:upper-1])
ax.plot(range(0,upper-lower),p(range(0,upper-lower)))
#ax.plot(range(0,upper-lower),rang_eth_data[lower-1:upper-1])

fig2, ax_heikin = plt.subplots()
candlestick2_ohlc(ax_heikin,open_heikin[lower:upper],high_heikin[lower:upper],low_heikin[lower:upper],close_heikin[lower:upper],width=0.6)
#ax_heikin.plot(range(0,upper-lower),fun(range(0,upper-lower),*popt))
ax_heikin.plot(range(0,upper-lower),heikin_rm_eth_data[lower-1:upper-1])
ax_heikin.plot(range(0,upper-lower),lower_bollinger_bands_r['Open'][lower-1:upper-1])
ax_heikin.plot(range(0,upper-lower),upper_bollinger_bands_r['Open'][lower-1:upper-1])

fig3, ax_poly_errors = plt.subplots()
ax_poly_errors.plot(range(1,11),Errors)
#plt.subplot(2,1,1)
#plt.ylabel("Opening price [$]")
#plt.title('ETH')
#plt.legend(loc=0)

#plt.subplot(2,1,2)
#plt.xlabel("Time in hours")
#plt.ylabel("Difference [$]")

#plt.savefig('ETH',dpi=1000)
plt.show()