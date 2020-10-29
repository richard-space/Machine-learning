# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:12:42 2019

@author: 益慶
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#使用內建鳶尾花資料
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#隨機創建資料
#X = np.random.rand(200,2)
#X=np.array([[1 ,1.1 ,1.2 ,1.3 ,1.4 ,1.32 ,2  , 2.1, 2.3 ,2.4, 6.1, 6.2, 6.3, 6.21],
#       [2 ,1.9 ,1.8 ,1.7 ,1.74 ,2.2, 5.2, 5.3 ,5.3, 5.5 ,-0.2 ,-0.23, -0.3, -0.5]])
#X=X.T

#使用sigmoid計算weight
def sigmoid(x):
    return 1/(1+np.exp(-x))

def WKmeans(data,k,err):
    step=1
    stop_control=1
    sse=0
    center=data[random.sample(range(0,np.size(data,0)),k)]
    new_center=np.zeros((k,np.size(data,1)))
    distance=np.zeros((k,np.size(data,0)))
    
    while step<500 and stop_control>err:   
        for j in range(k):
            weight=sigmoid(data-center[j])
            distance[j]=np.true_divide(np.sum(np.multiply(weight,np.square(data-center[j])),axis=1),np.sum(weight))
        group=np.argmin(distance,axis=0)
        for q in range(k):
            Idx=np.where(group==q)
            idata=[data[a] for a in Idx]            
            new_center[q]=np.mean(idata,axis=1)
            sse=sse+np.sum(np.square(idata-center[q]))                     
        stop_control=np.sum(np.abs(center-new_center))
        center=new_center
        step=step+1

    return(center,sse,group)
    
SSE=[]       
for k in range(1,8):      
    c,d,g=WKmeans(X,k,10**(-5))
    SSE.append(d)
    color=['b','g','r','c','m','y','orange']
    for p in range(np.size(X,0)):
        for q in range(k):
            if g[p]==q:
                plt.scatter(X[p,0],X[p,1],color=color[q])
    plt.scatter(c[:,0],c[:,1],marker='x',color='k',s=100)    
    plt.show()
plt.plot(range(1,8),SSE)
plt.xlabel('numbers of k')
plt.ylabel('sum of square')
plt.title('Elbow Method')
plt.show()

for p in range(np.size(X,0)):
        for q in range(3):
            if Y[p]==q:
                plt.scatter(X[p,0],X[p,1],color=color[q])
plt.xlabel('sepal length')  
plt.ylabel('sepal width') 
plt.title('iris true result')
plt.show()

