# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Reading the dataset from a csv file using pandas in the dataframe df. In df the training examples are stacked in differnet rows. The first column contains labels while the other columns contains pixel intesities.
# X is the input matrix containg different training examples satcked in different columns.
# Y_orig is a row matrix containg labels for training examples
# Y contains labels with one hot ebcoding
df = pd.read_csv('mnist_train.csv')
data = np.array(df)
X = (data[:,1:].transpose())/255
m = X.shape[1]
n = X.shape[0]
Y_orig = data[:,0:1].transpose()
Y = np.zeros((10,m))
for i in range(m): 
    Y[int(Y_orig[0,i]),i] = 1


# In[3]:


# The relu function is used to add non linearity. It is defined as relu(z) = max(z,0) or relu(z) = (z + |z|)/2
def relu(Z):
    result = (Z + np.abs(Z))/2
    return result


# In[4]:


# This function finds (d/dz)(relu(z)) which is equal to relu(z)/z. This function is useful during backpropagation. 
def relu_backward(Z):
    result = (Z + np.abs(Z))/(2*Z)
    return result


# In[5]:


# Softmax function is used to calculate the probabilities of the input belonging to each class. 
def softmax(Z):
    temp = np.exp(Z)
    result = temp/np.sum(temp,axis = 0,keepdims = True)
    return result


# In[6]:


# This function initializes all the parameters and store them in the dictionary named parameters
# The weights are initialized to random numbers from a gaussian distribution having mean 0 and Standatd deviation 0.01. 
# The biases are initialized to 0
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) - 1
    for l in range(1,L + 1):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
        #print(parameters)
    return parameters


# In[7]:


# The function is used to move forward through the network.
# For all the layers except the final one the function first calculates Z(l) = W(l)*A(l-1) + b(l). Then adds a relu non linearity using A(l) = relu(Z(l)). Here l denotes the layer number.
# For the final layer L the function first calculates Z(L) = W(L).A(L-1) + b(L), then uses softmax function to calculate the probabilities of the input belonging to each class. 
# The function saves the values of Z for all the layers in the dictionary named cache.
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


# In[8]:


# This function calculates the CROSS ENTROPY LOSS of our prdiction. The loss is defined as loss(a,y) = sum(-y(i)*log(a(i))).    
def compute_cost(AL,Y):
    m = AL.shape[1]
    cost = (np.sum(-(Y * np.log(AL))))/(m)
    return cost


# In[9]:


# This function calculates the derivatives of the loss with respect to the parameters. 
# Lets denote the derivative of the loss with respect to var as dvar. 
# It can be shown that for the final layer L dZ(L) = A(L) - Y(L) and for other layers dZ(l) = (W(l+1)transpose . dZ(l+1)) * A(l) for l belonging to (1,L-1).
# After finding dZ, dW and db can be found using dW(l) = dZ(l) . A(l-1)transpose and db(l) = sum over all the columns of dZ(l).
def backward_prop(X,Y,cache,parameters,AL,layer_dims):
    m = X.shape[1]
    dparameters = {}
    L = len(layer_dims) - 1
    dZ = AL - Y
    dparameters["dW" + str(L)] = dZ.dot(relu(cache["Z" + str(L-1)]).transpose())/m
    #dparameters["dW" + str(L)] = dZ.dot(X.transpose())/m
    dparameters["db" + str(L)] = np.sum(dZ,axis = 1,keepdims = True)/m
    for l in range(1,L):
        dZ = ((parameters["W" + str(L-l+1)].transpose()).dot(dZ)) * (relu_backward(cache["Z" + str(L-l)]))
        if L-l-1 != 0:
            dparameters["dW" + str(L-l)] = dZ.dot(relu(cache["Z" + str(L-1-l)]).transpose())/m
        else:
            dparameters["dW" + str(L-l)] = dZ.dot(X.transpose())/m
        dparameters["db" + str(L-l)] = np.sum(dZ,axis = 1,keepdims = True)/m
    return dparameters  


# In[10]:


# The parameters are updated using simple update rule W := W - learning_rate*dW and b := b - learning_rate*db
def update_parameters(parameters,dparameters,layer_dims,learning_rate):
    L = len(layer_dims) - 1
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*dparameters["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*dparameters["db" + str(l)]
    return parameters


# In[11]:


# The model function combines al the above functions.
#Model: Forward pass to calculate the prediction --> calcultes loss --> Calcultes derivatives of the loss with respect to the parameters --> Updates the parameters to minimize the loss
def model(X,Y,layer_dims,learning_rate,num_iters):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(num_iters):
        AL,cache = forward_prop(X,parameters)
        cost = compute_cost(AL,Y)
        costs.append(cost)
        dparameters = backward_prop(X,Y,cache,parameters,AL,layer_dims)
        parameters = update_parameters(parameters,dparameters,layer_dims,learning_rate)
        print(i,"\t",cost)
    return parameters,costs
    print(len(costs))


# In[12]:


# Trainig the model by choosing architecture of the model and hyperparameters
layer_dims = [784,120,10]
parameters,costs = model(X,Y,layer_dims,0.1,2000)


# In[13]:


# Plotting the cost to make sure it is decreasing
plt.plot(costs)


# In[14]:


# Saving the paramters found after training in the hard disc in csv format
dfW1 = pd.DataFrame(parameters["W1"])
dfW1.to_csv('Trained Parameters/W1.csv',header = None,index = None)
dfW2 = pd.DataFrame(parameters["W2"])
dfW2.to_csv('Trained Parameters/W2.csv',header = None,index = None)
dfb1 = pd.DataFrame(parameters["b1"])
dfb1.to_csv('Trained Parameters/b1.csv',header = None,index = None)
dfb2 = pd.DataFrame(parameters["b2"])
dfb2.to_csv('Trained Parameters/b2.csv',header = None,index = None)

