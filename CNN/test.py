# In[1]:


import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


# In[2]:


# Creating the model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,5)
        self.fc1 = nn.Linear(4*4*64,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        a1 = self.pool(F.relu(self.conv1(x)))
        a2 = self.pool(F.relu(self.conv2(a1)))
        a2 = a2.view(a2.shape[0],-1)
        a3 = F.relu(self.fc1(a2))
        a4 = (self.fc2(a3))
        return a4


# In[3]:


# Creating an instance of the model and loading the parameters
net = Net()
net.load_state_dict(tc.load('parameters.pth'))


# In[4]:


# Loading the test dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False,download=True,transform = transform)
testloader = tc.utils.data.DataLoader(testset, batch_size=10000,shuffle=True, num_workers=2)


# In[5]:


# Calculating the test accuracy
for data1 in testloader:
    test_inputs, test_labels = data1
    #print(test_inputs.shape)
    test_outputs = net(test_inputs)
predict = np.zeros(test_outputs.shape[0])
accuracy = 0
for i in range(test_outputs.shape[0]):
    max = test_outputs[i,0]
    index = 0
    for j in range(1,test_outputs.shape[1]):
        if test_outputs[i,j] > max:
            max = test_outputs[i,j]
            index = j
    predict[i] = index
    if int(predict[i]) == int(test_labels[i]):
        accuracy = accuracy + 1
print('Accuracy =',(accuracy/test_outputs.shape[0])*100) 







