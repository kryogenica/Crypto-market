import numpy as np
my_data = np.genfromtxt('sentimental.txt',delimiter=',')

none_zero=[]

for i in range(0,my_data.shape[0]):
    if (my_data[i]>0.0):
        none_zero.append(my_data[i])

none_zero=np.asanyarray(none_zero)
Mean=none_zero.mean()

for i in range(0,my_data.shape[0]):
    if (float(my_data[i])==float(0.0)):
        my_data[i]=Mean
print (my_data)

np.savetxt("sentimental_arranged.txt", my_data, delimiter=",")

