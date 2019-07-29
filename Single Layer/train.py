# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Reading the dataset from a csv file using pandas in the dataframe df. In df the training examples are stacked in differnet rows. The first column contains labels while the other columns contains pixel intesities.
# X is the input matrix containg different training examples satcked in different columns.
# Y_orig is a row matrix containg labels for training examples
# Y contains labels with one hot ebcoding
df = pd.read_csv('mnist_train.csv',header = None)
data = np.array(df)
m = data.shape[0]
X = (data[:,1:].transpose())/255
n = X.shape[0]
Y_orig = data[:,0:1].transpose()
Y = np.zeros((10,m))
for i in range(m): 
    Y[int(Y_orig[0,i]),i] = 1


# In[3]:


# Setting the values of Hyperparameters 
learning_rate = 0.1
num_iters = 2000


# In[4]:


# Initializing W and b to random numbers and zero respectively
W = np.random.randn(10,n)*0.01
b = np.zeros((10,1))

# The list costs stores value of cost after every iteration
costs = []

for i in range(num_iters):
    # Finding Z 
    Z = W.dot(X) + b
    # Calculating A using A = softmax(Z) 
    temp = np.exp(Z) 
    A = temp/(np.sum(temp,axis = 0,keepdims = True))
    # Calculating the CROSS ENTROPY LOSS of our prdiction. The loss is defined as loss(a,y) = sum(-y(i)*log(a(i))
    cost = (np.sum(-(Y * np.log(A))))/m
    # Printing loss after every 50 iterations
    if i%50 == 0 :
        print(i,"\t",cost)
    costs.append(cost)
    # Lets denote the derivative of loss with respect to var as dvar. It can be then shown that dZ = A -Y 
    dZ = A - Y
    # Calculating dW and db using previously found dZ
    dW = dZ.dot(X.transpose())/m
    db = np.sum(dZ,axis = 1,keepdims = True)/m
    # Updating the parameters
    W = W - learning_rate*dW
    b = b - learning_rate*db


# In[5]:


# Plotting the cost to make sure it is decreasing
plt.plot(costs)


# In[7]:


# Saving the paramters found after training in the hard disc in csv format. 
dfW = pd.DataFrame(W)
dfW.to_csv('Trained Parameters/W.csv',header = None,index = None)
dfb = pd.DataFrame(b)
dfb.to_csv('Trained Parameters/b.csv',header = None,index = None)

