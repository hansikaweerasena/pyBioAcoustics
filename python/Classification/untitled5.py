# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 12:51:55 2017

@author: Hansika
"""
from sklearn import svm
import pandas as pd
import pickle
import numpy as np
import collections
from sklearn.ensemble import IsolationForest

#Fs, y = readSegment('G:\FYP Project\Final Evaluation\\NEW\\0\\human_seg.wav')
#test_seg = getFeatureForSegment(y,Fs)
def makeTrainCSV():
    CSaTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\CSaTrainMeanSDFeatures.csv")
    ECTrain =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\ECTrainMeanSDFeatures.csv")
    CSnTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\CSnTrainMeanSDFeatures.csv")
    BWSTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\BWSTrainMeanSDFeatures.csv")
    EOTrain = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\train\EOTrainMeanSDFeatures.csv")
    
    return pd.concat([CSaTrain, ECTrain,CSnTrain,EOTrain,BWSTrain])

def makeTestCSV():
    CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\CSaTestMeanSDFeatures.csv")
    ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\ECTestMeanSDFeatures.csv")
    CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\CSnTestMeanSDFeatures.csv")
    EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\EOTestMeanSDFeatures.csv")
    BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\BWSTestMeanSDFeatures.csv")
    
    return pd.concat([CSaTest, ECTest,CSnTest,EOTest,BWSTest])

test_new_dataframe = makeTestCSV()
test_new_dataframe = test_new_dataframe.reset_index(drop=True)
train_dataframe = makeTrainCSV()
train_features = train_dataframe.iloc[:,:-1]
#clf = svm.OneClassSVM(nu=0.05, kernel="rbf", degree=3,gamma = 0.9)
clf = IsolationForest(max_samples=100)
#clf = svm.OneClassSVM(nu=0.05, kernel="rbf",gamma = 0.6)
clf.fit(train_features)
test_features = test_new_dataframe.iloc[:,:-1]
test_features = np.array(test_features)
pred_test = clf.predict(test_features)
#pred_test = clf.predict(test_seg)
print 'predicted as',pred_test

filename = 'outlier_detector.sav'
pickle.dump(clf, open(filename, 'wb'))
print collections.Counter(pred_test)
j, = np.where( pred_test==-1 )
new_test=test_new_dataframe.drop(test_new_dataframe.index[j])
classfiy(new_test, classifier,'All')