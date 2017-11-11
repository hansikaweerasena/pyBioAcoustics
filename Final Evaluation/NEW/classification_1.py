# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:43:35 2017

@author: Hansika
"""
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

def makeTrainCSV():
    CSaTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\train\CSaTrainMeanSDFeatures.csv")
    ECTrain =  pd.read_csv("G:\FYP Project\GIT project\Features\\train\ECTrainMeanSDFeatures.csv")
    CSnTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\train\CSnTrainMeanSDFeatures.csv")
    BWSTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\train\BWSTrainMeanSDFeatures.csv")
    EOTrain = pd.read_csv("G:\FYP Project\GIT project\Features\\train\EOTrainMeanSDFeatures.csv")
    
    return pd.concat([CSaTrain, ECTrain,EOTrain,BWSTrain])

def makeTestCSV():
    CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\CSaTestMeanSDFeatures.csv")
    ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\test\ECTestMeanSDFeatures.csv")
    CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\CSnTestMeanSDFeatures.csv")
    EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\test\EOTestMeanSDFeatures.csv")
    BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\BWSTestMeanSDFeatures.csv")
    
    return pd.concat([CSaTest, ECTest,EOTest,BWSTest])

def saveModel(model):
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    
def classfiy(test_dataframe,  classifier , name):
    print "for ", name 
    test_labels = test_dataframe.label
    #labels = list(set(test_labels))
    test_labels = np.array([labels.index(x) for x in test_labels])
    
    test_features = test_dataframe.iloc[:,:-1]
    test_features = np.array(test_features)

    results = classifier.predict(test_features)
    num_correct = (results == test_labels).sum()
    recall = num_correct *1.0/ len(test_labels)
    print "model accuracy (%): ", recall * 100, "%"
    print "Num correct", num_correct, "Total", len(test_labels)
    print classifier.score(test_features,test_labels)
    return results 

#train_dataframe = pd.read_csv("G:\FYP Project\GIT project\Features\\train\TrainMeanSDFeatures.csv")
train_dataframe = makeTrainCSV()
train_labels = train_dataframe.label
labels = list(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = train_dataframe.iloc[:,:-1]


classifier = svm.SVC()
classifier.fit(train_features, train_labels)

test_dataframe = makeTestCSV()
#test_dataframe = pd.read_csv("G:\FYP Project\GIT project\Features\\test\TestMeanSDFeatures.csv")
CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\CSaTestMeanSDFeatures.csv")
ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\test\ECTestMeanSDFeatures.csv")
CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\CSnTestMeanSDFeatures.csv")
EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\test\EOTestMeanSDFeatures.csv")
BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\\test\BWSTestMeanSDFeatures.csv")
#test_dataframe = makeTestCSV()

classfiy(CSaTest, classifier,'CSa')
classfiy(ECTest, classifier,'EC')
#classfiy(CSnTest, classifier, 'CSn')
classfiy(EOTest, classifier, 'EO')
classfiy(BWSTest, classifier, 'BWS')
classfiy(test_dataframe, classifier,'All')