#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:14:51 2018

@author: Jake
"""

'''
Exercise 4.4.2 from edX course DL0110EN:
Deep Learning with Python and PyTorch

Deeper Neural Networks with nn.ModuleList()
'''

# Import libraries, set seed
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import ListedColormap
torch.manual_seed(1)

# Plot function copied from lab directly
def plot_decision_regions_3class(model,data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
    X=data_set.x.numpy()
    y=data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min()-0.1 , X[:, 0].max()+0.1 
    y_min, y_max = X[:, 1].min()-0.1 , X[:, 1].max() +0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    XX=torch.torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _,yhat=torch.max(model(XX),1)
    yhat=yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:]==0,0],X[y[:]==0,1],'ro',label='y=0')
    plt.plot(X[y[:]==1,0],X[y[:]==1,1],'go',label='y=1')
    plt.plot(X[y[:]==2,0],X[y[:]==2,1],'o',label='y=2')
    plt.title("decision region")
    plt.legend()
    plt.show()

# Data set copied from lab directly - spiral dataset
class Data(Dataset):
    #  modified from: http://cs231n.github.io/neural-networks-case-study/
    def __init__(self,K=3,N=500):
        D = 2
        X = np.zeros((N*K,D)) # data matrix (each row = single example)
        y = np.zeros(N*K, dtype='uint8') # class labels
        for j in range(K):
          ix = range(N*j,N*(j+1))
          r = np.linspace(0.0,1,N) # radius
          t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
          X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
          y[ix] = j
    
        self.y=torch.from_numpy(y).type(torch.LongTensor)
        self.x=torch.from_numpy(X).type(torch.FloatTensor)
        self.len=y.shape[0]
        
    def __getitem__(self,index):    
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    
    def plot_stuff(self):
        plt.plot(self.x[self.y[:]==0,0].numpy(),self.x[self.y[:]==0,1].numpy(),'o',label="y=0")
        plt.plot(self.x[self.y[:]==1,0].numpy(),self.x[self.y[:]==1,1].numpy(),'ro',label="y=1")
        plt.plot(self.x[self.y[:]==2,0].numpy(),self.x[self.y[:]==2,1].numpy(),'go',label="y=2")
        plt.legend()
        plt.show()

class Net(nn.Module):
    def __init__(self,Layers):
        super(Net,self).__init__()
        self.hidden = nn.ModuleList()

        for input_size,output_size in zip(Layers,Layers[1:]):
            self.hidden.append(nn.Linear(input_size,output_size))
        
    def forward(self,activation):
        L=len(self.hidden)
        for (l,linear_transform)  in zip(range(L),self.hidden):
            if l<L-1:
                activation =F.relu(linear_transform (activation))
           
            else:
                activation =linear_transform (activation)
        
        return activation

def train(data_set,model,criterion, train_loader, optimizer, epochs=100):
    LOSS=[]
    ACC=[]
    for epoch in range(epochs):
        for x,y in train_loader:
            optimizer.zero_grad()
        
            yhat=model(x)
            loss=criterion(yhat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        LOSS.append(loss.item())
        ACC.append(accuracy(model,data_set))
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS,color=color)
    ax1.set_xlabel('epoch',color=color)
    ax1.set_ylabel('total loss',color=color)
    ax1.tick_params(axis='y', color=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot( ACC, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()
    return LOSS

def accuracy(model,data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()

# Create dataset object
data_set = Data()
data_set.plot_stuff()
data_set.y = data_set.y.view(-1)

# Network constants - kept constant among networks for comparison
learning_rate = 0.01
epochs = 1000
criterion = nn.CrossEntropyLoss()

# Data & train loaders
train_loader = DataLoader(dataset = data_set, batch_size = 20)

# Create network w/ 1 hidden layer, 50 neurons, 3 classes
Layers1 = [2,50,3]
model1 = Net(Layers1)

optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
LOSS1 = train(data_set, model1, criterion,
              train_loader, optimizer1, epochs = epochs)

plot_decision_regions_3class(model1, data_set)

# Create network with 2 hidden layers, 20 neurons, 3 classes
Layers2 = [2,10,10,3]
model2 = Net(Layers2)

optimizer2 = torch.optim.SGD(model2.parameters(), lr = learning_rate)
LOSS2 = train(data_set, model2, criterion,
              train_loader, optimizer2, epochs = epochs)
plot_decision_regions_3class(model2, data_set)

# Practice question:
# create a network with three hidden layers each with ten neurons, then train
# the network using the same process as above

print('\n -------------' )
print('| My Solution |')
print(' -------------')

Layers3 = [2, 10, 10, 10, 3]
model3 = Net(Layers3)

optimizer3 = torch.optim.SGD(model3.parameters(), lr = learning_rate)
LOSS3 = train(data_set, model3, criterion,
              train_loader, optimizer3, epochs = epochs)
plot_decision_regions_3class(model3, data_set)

'''
Double-click __here__ for the solution.

<!-- 
Layers=[2,10,10,10,3]
model=Net(Layers)
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader=DataLoader(dataset=data_set,batch_size=20)
criterion=nn.CrossEntropyLoss()
LOSS=train(data_set,model,criterion, train_loader, optimizer, epochs=1000)
plot_decision_regions_3class(model,data_set
-->
'''