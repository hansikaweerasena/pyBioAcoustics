# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:43:35 2017

@author: Hansika
"""
import pandas as pd
import numpy as np
from sklearn import svm

def makeTrainCSV():
    CSaTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\train\CSaTrainMeanSDFeatures.csv")
    ECTrain =  pd.read_csv("G:\FYP Project\GIT project\Features\\train\ECTrainMeanSDFeatures.csv")
    
    return pd.concat([CSaTrain, ECTrain])

def makeTestCSV():
    CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\CSaTestMeanSDFeatures.csv")
    ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\test\ECTestMeanSDFeatures.csv")
    
    return pd.concat([CSaTest, ECTest])


train_dataframe = makeTrainCSV()
train_labels = train_dataframe.label
labels = list(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = train_dataframe.iloc[:,:-1]


classifier = svm.SVC()
classifier.fit(train_features, train_labels)


test_dataframe = makeTestCSV()

test_labels = test_dataframe.label
labels = list(set(test_labels))
test_labels = np.array([labels.index(x) for x in test_labels])

test_features = test_dataframe.iloc[:,:-1]
test_features = np.array(test_features)

results = classifier.predict(test_features)
num_correct = (results == test_labels).sum()
recall = num_correct *1.0/ len(test_labels)
print "model accuracy (%): ", recall * 100, "%"