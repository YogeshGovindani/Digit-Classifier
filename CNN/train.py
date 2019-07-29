# In[1]:


import torch as tc
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


# Loading Dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True,transform = transform)
trainloader = tc.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)


# In[3]:


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


# In[4]:


costs = []


# In[5]:


# Creating an instance of the moedel class
net = Net()


# In[6]:


# Defining the optimizer and the cost function.The learning was manually decreased by a factor of 10 after every 2 epochs
compute_cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),0.001)


# In[7]:


# Training the model
for epoch in range(1):
    for i,data in enumerate(trainloader, 0):
        inputs,labels = data
        outputs = net(inputs)
        loss = compute_cost(outputs, labels)
        print(i,'\t',loss.item())
        costs.append(loss)
        loss.backward()
        optimizer.step()


# In[8]:


# Plotting the Cost with number of iterations
plt.plot(costs)


# In[19]:


# Saving the model parameters
tc.save(net.state_dict(), 'parameters.pth')








