# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:59:26 2018

@author: 张星光
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat
import seaborn as sns

os.chdir('E:\\S\\ECE629\\Weekly\\10')
in_data = loadmat('iris.mat')

class processing_data:
    X, y, eigval, MsErr, W, b, new_dim, mean_X, mse = 0,0,0,0,0,0,0,0,0
data = processing_data
data.X = in_data['X']
data.y = in_data['y']

#def pca(self,arg1):
arg1 = 2
dim,num_data = np.shape(data.X);
mean_X = np.transpose(np.mean(np.transpose(data.X),0))
X = data.X - np.reshape(mean_X,(dim,1)) * np.ones((1,num_data))
S = np.matmul(X,np.transpose(X))

Cov = tf.placeholder(tf.float32, shape = S.shape) 
Sigma,_,V = tf.svd(Cov)
with tf.Session() as sess:
    data.eigval,VV = sess.run([Sigma,V], feed_dict={Cov:S})
    
data.eigval = data.eigval/num_data
sum_eig = np.matmul(np.triu(np.ones((dim,dim)),1),data.eigval)
data.MsErr = sum_eig

if arg1 >= 1:
  new_dim = arg1   
  # = find(sum_eig/sum(model.eigval) <= arg1);
  #if isempty(inx), new_dim = dim; else new_dim = inx(1); end
  #model.var = arg1;

data.W = -VV[:,0:new_dim]
data.b = -np.matmul(np.transpose(data.W),mean_X)
data.new_dim = new_dim
data.mean_X = mean_X
data.mse = data.MsErr[new_dim-1]

Out = data
Out.X = np.matmul(np.transpose(data.W),data.X) + np.reshape(data.b,(new_dim,1)) * np.ones((1,num_data))

color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime'], 2: sns.xkcd_rgb['ochre']}
#colors = list(map(lambda x: color_mapping[x], Out.y.tolist()))

plt.figure(1,figsize=(12,9))
for i in range(num_data):
    plt.scatter(Out.X[0,i],Out.X[1,i],c=color_mapping[data.y[0,i]-1])