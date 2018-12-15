# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:10:01 2018

@author: Xingguang Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns

in_data1 = loadmat('riply_trn.mat')
in_data2 = loadmat('riply_tst.mat')
class data:
    X,y,dim,num_data = 0,0,0,0
class model:
    W,b,separab = 0,0,0
    
train = data;
train.X = in_data1['X']
train.y = in_data1['y']
train.dim, train.num_data = np.shape(train.X)
test_X = in_data2['X']
test_y = in_data2['y']

inx1 = np.argwhere(train.y == 1)
inx2 = np.argwhere(train.y == 2)
n1 = np.size(inx1,0)
n2 = np.size(inx2,0)
m1 = np.mean(train.X[:,inx1[:,1]],1)
m2 = np.mean(train.X[:,inx2[:,1]],1)
norm_x1 = train.X[:,inx1[:,1]] - np.reshape(m1,[train.dim,1]) * np.ones((1,n1))
norm_x2 = train.X[:,inx2[:,1]] - np.reshape(m2,[train.dim,1]) * np.ones((1,n2))

S1 = np.matmul(norm_x1, np.transpose(norm_x1))
S2 = np.matmul(norm_x2, np.transpose(norm_x2))
Sw = S1 + S2

W = np.matmul(np.linalg.inv(Sw), (m1-m2))
proj_m1 = np.matmul(np.transpose(W),m1)
proj_m2 = np.matmul(np.transpose(W),m2)
model.W = W
model.b = -0.5*(proj_m1+proj_m2);
proj = np.matmul(np.matmul(np.transpose(W),Sw),W)
model.separab = (proj_m1-proj_m2)**2/proj

#test
dim, num_data = np.shape(test_X)
dfce = np.matmul(np.transpose(model.W),test_X) + model.b
y = np.ones((1,num_data))
idx = np.argwhere(dfce <0)
y[0,idx] = 2
y1 = np.reshape(y,[1,-1])
y2 = np.reshape(test_y,[1,-1])
e = np.size(np.argwhere(y1 - y2 != 0),0) / num_data

#plot
color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['ochre']}
colors = list(map(lambda x: color_mapping[x], train.y[0,:]-1))
X = [-1.3,1]
Y = X
Y[0] = -(W[0]*X[0]+model.b)/W[1]
Y[1] = -(W[0]*X[1]+model.b)/W[1]
plt.figure(1,figsize=(12,9))
plt.scatter(train.X[0,:],train.X[1,:],c=colors)
plt.plot(X,Y)
