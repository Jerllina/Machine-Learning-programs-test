# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:11:02 2018

@author: lijie
"""
import numpy as np
import os
import matplotlib.pyplot as plt

'''load data'''
def load_data(filename):
    dataset=[]
    labels=[]
    with open(filename,'r') as f:
        for line in f:
            splited_line=[float(i) for i in line.strip().split('\t')]
            data=[1.0]+splited_line[:-1]
            label=splited_line[-1]
            dataset.append(data)
            labels.append(label)
    dataset=np.array(dataset)
    labels=np.array(labels)
    return dataset,labels

'''Use gradient ascent algorithm to create classifier'''
class LogisticGradAscentOptimized_Classifier():
    #construct Sigmoid(x)
    @staticmethod
    def sigmoid(x):
        return 1.0/(1+np.exp(-x))

    #implement gradient ascent algorism
    def gradient_ascent(self,dataset,labels,max_iter=10000):
        dataset=np.matrix(dataset)
        vlabels=np.matrix(labels).reshape(-1,1)
        m,n=dataset.shape
        w=np.ones((n,1))
        alpha=0.001
        ws=[]
        for i in range(max_iter):
            error=vlabels-self.sigmoid(dataset*w)
            w+=alpha*(dataset.T)*error
            ws.append(w.reshape(1,-1).tolist()[0])
        
        self.w=w
        return w,np.array(ws)

    #implement LR classifier
    def LRclassifier(self,data,w=None):
        if w is None:
            w=self.w
        
        data=np.matrix(data)
        prob=self.sigmoid((data*w).tolist()[0][0])
        return round(prob)
    
'''visualize the decision boundary'''
def snapshot(w,dataset,labels,pic_name):
    #plot the whole data picture
    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')
        
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    pts={}
    for data,label in zip(dataset.tolist(),labels.tolist()):
        pts.setdefault(label,[data]).append(data)
    for label,data in pts.items():
        data=np.array(data)           
        plt.scatter(data[:,1],data[:,2],label=label,alpha=0.5)

    #plot the decision boundary line
    def pred_y(x,w):
        w0,w1,w2=w
        return (-w0-w1*x)/w2
        
    x=[-4.0,3.0]
    y=[pred_y(i,w) for i in x]
    
    plt.plot(x,y,linewidth=2,color='#FB4A42')
    
    pic_name='./snapshots/{}'.format(pic_name)
    fig.savefig(pic_name)
    plt.close(fig)
    
if'__main__' == __name__:
    clf = LogisticGradAscentOptimized_Classifier()
    dataset,labels =load_data('testSet.txt')
    w,ws = clf.gradient_ascent(dataset,labels,max_iter=50000)
    m,n = ws.shape
    # plot the split line
    for i in range(300):
        if i%(30)==0:
            print('{}.png saved'.format(i))
            snapshot(ws[i].tolist(),dataset,labels,'{}.png'.format(i))
    
    fig = plt.figure()
    for i in range(n):
        label='w{}'.format(i)
        ax=fig.add_subplot(n,1,i+1)
        ax.plot(ws[:,i],label=label)
        ax.legend()
        
    fig.savefig('w_traj.png')
        
    
    
        
            
            
