# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Getting W and b from the csv files using pandas  
dfW = pd.read_csv('Trained Parameters/W.csv',header = None)
W = np.array(dfW)
dfb = pd.read_csv('Trained Parameters/b.csv',header = None)
b = np.array(dfb)


# In[3]:


# Calculating test accuracy
df = pd.read_csv('mnist_test.csv',header = None)
data = np.array(df)
X_test = (data[:,1:].transpose())/255
Y_test = data[:,0:1].transpose()
accuracy = 0
m_test = X_test.shape[1]
predict = np.zeros((1,m_test))
Z = W.dot(X_test) + b
temp = np.exp(Z) 
A_test = temp/(np.sum(temp,axis = 0,keepdims = True))
for i in range(m_test):
    max = 0
    for j in range(10):
        if A_test[j,i] > max:
            max = A_test[j,i]
            max_index = j
        predict[0,i] = max_index
    if predict[0,i] == Y_test[0,i]:
        accuracy = accuracy + 1
print(accuracy/X_test.shape[1] * 100,'%')





