# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Getting W and b from the csv files using pandas  
parameters = {}
dfW1 = pd.read_csv('Trained Parameters/W1.csv',header = None)
parameters["W1"] = np.array(dfW1)
dfW2 = pd.read_csv('Trained Parameters/W2.csv',header = None)
parameters["W2"] = np.array(dfW2)
dfb1 = pd.read_csv('Trained Parameters/b1.csv',header = None)
parameters["b1"] = np.array(dfb1)
dfb2 = pd.read_csv('Trained Parameters/b2.csv',header = None)
parameters["b2"] = np.array(dfb2)


# In[3]:


def relu(Z):
    result = (Z + np.abs(Z))/2
    return result


# In[4]:


def softmax(Z):
    temp = np.exp(Z)
    result = temp/np.sum(temp,axis = 0,keepdims = True)
    return result


# In[5]:


def forward_prop(X,parameters):
    cache = {}
    L = len(layer_dims) - 1
    A_prev = X
    for l in range(1,L):
        Z = parameters["W" + str(l)].dot(A_prev) + parameters["b" + str(l)]
        A = relu(Z)
        cache["Z" + str(l)] = Z
        A_prev = A
    Z = parameters["W" + str(L)].dot(A_prev) + parameters["b" + str(L)]
    AL = softmax(Z)
    cache["Z" + str(L)] = Z
    return AL,cache


# In[6]:


# Calculating test accuracy
layer_dims = [784,120,10]
df = pd.read_csv('mnist_test.csv',header = None)
data = np.array(df)
X_test = (data[:,1:].transpose())/255
Y_test = data[:,0:1].transpose()
accuracy = 0
m_test = X_test.shape[1]
predict = np.zeros((1,m_test))
A_test,cache = forward_prop(X_test,parameters)
for i in range(m_test):
    max = 0
    for j in range(10):
        if A_test[j,i] > max:
            max = A_test[j,i]
            max_index = j
        predict[0,i] = max_index
    if predict[0,i] == Y_test[0,i]:
        accuracy = accuracy + 1
accuracy = (accuracy/m_test)*100
print(accuracy,"%")



