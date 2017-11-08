# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:43:35 2017

@author: Hansika
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

def calculateCrossValidateScore(kernel,train_features, train_labels):
    clf = svm.SVC(kernel = kernel, degree = 4)
    scores = cross_val_score(clf, train_features, train_labels, cv=10)
    print "mean Cross Validated Score is : ", scores.mean()

def makeTrainCSV():
    CSaTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\train\CSaTrainTWMeanSDFeatures.csv")
    ECTrain =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\train\ECTrainTWMeanSDFeatures.csv")
    CSnTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\train\CSnTrainTWMeanSDFeatures.csv")
    BWSTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\train\BWTrainTWMeanSDFeatures.csv")
    EOTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\train\EOTrainTWMeanSDFeatures.csv")
    
    return pd.concat([CSaTrain, ECTrain,CSnTrain,EOTrain,BWSTrain])

def makeTestCSV():
    CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\CSaTestTWMeanSDFeatures.csv")
    ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\ECTestTWMeanSDFeatures.csv")
    CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\CSnTestTWMeanSDFeatures.csv")
    EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\EOTestTWMeanSDFeatures.csv")
    BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\BWTestTWMeanSDFeatures.csv")
    
    return pd.concat([CSaTest, ECTest,CSnTest,EOTest,BWSTest])

def classfiy(test_dataframe,  classifier2 , name):
    print "for ", name 
    test_labels = test_dataframe.label
    #labels = list(set(test_labels))
    test_labels = np.array([labels.index(x) for x in test_labels])
    
    test_features = test_dataframe.iloc[:,:-1]
    test_features = np.array(test_features)

    results = classifier2.predict(test_features)
    num_correct = (results == test_labels).sum()
    recall = num_correct *1.0/ len(test_labels)
    print "model accuracy (%): ", recall * 100, "%"
    print "Num correct", num_correct, "Total", len(test_labels)
    print classifier2.score(test_features,test_labels)
    return results 

#train_dataframe = pd.read_csv("G:\FYP Project\GIT project\Features\\train\TrainMeanSDFeatures.csv")
train_dataframe = makeTrainCSV()
train_labels = train_dataframe.label
labels = list(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = train_dataframe.iloc[:,:-1]

test_dataframe = makeTestCSV()
#test_dataframe = pd.read_csv("G:\FYP Project\GIT project\Features\\test\TestMeanSDFeatures.csv")
CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\CSaTestTWMeanSDFeatures.csv")
ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\ECTestTWMeanSDFeatures.csv")
CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\CSnTestTWMeanSDFeatures.csv")
EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\EOTestTWMeanSDFeatures.csv")
BWSTest2 = pd.read_csv("G:\FYP Project\\Two windows segments\Black-winged-Stilt\\Test\Segment\BWTestTWMeanSDFeatures.csv")
#test_dataframe = makeTestCSV()
for kernel in ('linear', 'poly', 'rbf'):
    print "-------------------------------------------------------------------------"
    print "kernal is: ",kernel
    classifier2 = svm.SVC(kernel=kernel,degree=4)
    classifier2.fit(train_features, train_labels)
    calculateCrossValidateScore(kernel,train_features, train_labels)
    
    classfiy(CSaTest, classifier2,'CSa')
    classfiy(ECTest, classifier2,'EC')
    classfiy(CSnTest, classifier2, 'CSn')
    classfiy(EOTest, classifier2, 'EO')
    classfiy(BWSTest2, classifier2, 'BWS')
    classfiy(test_dataframe, classifier2,'All')