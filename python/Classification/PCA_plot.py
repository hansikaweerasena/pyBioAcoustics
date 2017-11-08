# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 23:42:06 2017

@author: Hansika
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getPCA(test_data,n):
    pca = PCA(n_components=n)
    pca.fit(test_data)
    return pca
    
def makeTrainCSV():
    CSaTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\CSaTrainMeanSDFeatures.csv")
    ECTrain =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\ECTrainMeanSDFeatures.csv")
    CSnTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\CSnTrainMeanSDFeatures.csv")
    BWSTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\BWSTrainMeanSDFeatures.csv")
    EOTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\EOTrainMeanSDFeatures.csv")
    
    return pd.concat([CSaTrain, ECTrain,CSnTrain,EOTrain,BWSTrain])


train_dataframe = makeTrainCSV()
train_labels = train_dataframe.label
labels = list(set(train_labels))
y = np.array([labels.index(x) for x in train_labels])
train_features = train_dataframe.iloc[:,:-1]
pca = getPCA(train_features,2)
pca.fit(train_features)
transformed = pca.transform(train_features)

plt.scatter(transformed[y==0,0], transformed[y==0,1], label='BWS', c='red', marker = 'o')
plt.scatter(transformed[y==1,0], transformed[y==1,1], label='EO', c='blue', marker = '*')
plt.scatter(transformed[y==2,0], transformed[y==2,1], label='CSn', c='lightgreen',marker ='s')
plt.scatter(transformed[y==3,0], transformed[y==3,1], label='CSa', c='orange',marker ='^')
plt.scatter(transformed[y==4,0], transformed[y==4,1], label='EC', c='purple',marker ='v')
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.legend()
plt.savefig('G:\FYP Project\Graphs for paper\\2_PCA.eps', format='eps', dpi=1000)
plt.show()

pca = getPCA(train_features,3)
pca.fit(train_features)
transformed = pca.transform(train_features)

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(transformed[y==0,0], transformed[y==0,1], transformed[y==0,2], label='BWS', c='red', marker = 'o')
ax.scatter(transformed[y==1,0], transformed[y==1,1], transformed[y==1,2],label='EO', c='blue', marker = '*')
ax.scatter(transformed[y==2,0], transformed[y==2,1], transformed[y==2,2],label='CSn', c='lightgreen', marker = 's')
ax.scatter(transformed[y==3,0], transformed[y==3,1], transformed[y==3,2],label='CSa', c='orange', marker = '^')
ax.scatter(transformed[y==4,0], transformed[y==4,1], transformed[y==4,2],label='EC', c='purple', marker = 'v')
ax.legend()
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')
plt.savefig('G:\FYP Project\Graphs for paper\\3_PCA.eps', format='eps', dpi=1000)
plt.show()