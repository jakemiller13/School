#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 06:47:08 2018

@author: Jake
"""

'''
3.2 Training Logistic Regression
'''

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

class plot_error_surfaces(object):
    '''
    To help visualize data/parameter space during training
    Class supplied by lab
    '''
    def __init__(self,w_range, b_range,X,Y,n_samples=50,go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z=np.zeros((30,30))
        count1=0
        self.y=Y.numpy()
        self.x=X.numpy()
        for w1,b1 in zip(w,b):
            count2=0
            for w2,b2 in zip(w1,b1):
                
  
                yhat= 1 / (1 + np.exp(-1*(w2*self.x+b2)))
                Z[count1,count2]=-1*np.mean(self.y*np.log(yhat+1e-16) +(1-self.y)*np.log(1-yhat+1e-16))
                count2 +=1
    
            count1 +=1
        self.Z=Z
        self.w=w
        self.b=b
        self.W=[]
        self.B=[]
        self.LOSS=[]
        self.n=0
        if go==True:
            plt.figure()
            plt.figure(figsize=(7.5,5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    def get_stuff(self,model,loss):
        self.n=self.n+1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)
        
    def final_plot(self): 
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c='r', marker='x',s=200,alpha=1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W,self.B,c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x,self.y,'ro',label="training points")
        plt.plot(self.x,self.W[-1]*self.x+self.B[-1],label="estimated line")
        plt.plot(self.x,1 / (1 + np.exp(-1*(self.W[-1]*self.x+self.B[-1]))),label='sigmoid')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: '+str(self.n))
        plt.legend()
        plt.show()
        plt.subplot(122)
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W,self.B,c='r', marker='x')
        plt.title('Loss Surface Contour Iteration'+str(self.n) )
        plt.xlabel('w')
        plt.ylabel('b')
        plt.legend()

def PlotStuff(X,Y,model,epoch,leg=True):
    plt.plot(X.numpy(),model(X).detach().numpy(),label='epoch '+str(epoch))
    plt.plot(X.numpy(),Y.numpy(),'r')
    if leg==True:
        plt.legend()
    else:
        pass

class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-1,1,0.1).view(-1,1)
        self.y=-torch.zeros(self.x.shape[0],1)
        self.y[self.x[:,0]>0.2]=1
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

data_set=Data()
trainloader=DataLoader(dataset=data_set,batch_size=3)

class logistic_regression(nn.Module):
    def __init__(self,n_inputs):
        super(logistic_regression,self).__init__()
        self.linear=nn.Linear(n_inputs,1)
    def forward(self,x):
        yhat=torch.sigmoid(self.linear(x))
        return yhat

model=logistic_regression(1)

# Replace random initialized variable values
model.state_dict() ['linear.weight'].data[0]=torch.tensor([[-5]])
model.state_dict() ['linear.bias'].data[0]=torch.tensor([[-10]])

get_surface=plot_error_surfaces(15,13,data_set[:][0],data_set[:][1],30)

criterion = nn.BCELoss()
#def criterion(yhat,y):
#    out=-1*torch.mean(y*torch.log(yhat) +(1-y)*torch.log(1-yhat))
#    return out

learning_rate=2

optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(100):
    
    for x,y in trainloader:
        #make a prediction 
        yhat= model(x)
        #calculate the loss
        loss = criterion(yhat, y)
        #clear gradient
        optimizer.zero_grad()
        #Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        #the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        #for plotting
        get_surface.get_stuff(model,loss.tolist())
        #plot every 20 iterataions
    if epoch%20==0:
        get_surface.plot_ps()

yhat=model(data_set.x)
lable=yhat>0.5
print(torch.mean((lable==data_set.y.type(torch.ByteTensor)).type(torch.float)))



