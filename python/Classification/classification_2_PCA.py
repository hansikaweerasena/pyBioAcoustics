# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 17:09:58 2017

@author: Hansika
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

def calculateCrossValidateScore(kernel,train_features, train_labels):
    clf = svm.SVC(kernel = kernel,degree = 4)
    scores = cross_val_score(clf, train_features, train_labels, cv=10)
    print "mean Cross Validated Score is : ", scores.mean()
    return scores.mean()

def getPCA(test_data,n):
    pca = PCA(n_components=n)
    pca.fit(test_data)
    return pca
    
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


def classfiy(test_dataframe,  classifier , name,pca):
    print "for ", name 
    test_labels = test_dataframe.label
    #labels = list(set(test_labels))
    test_labels = np.array([labels.index(x) for x in test_labels])
    
    test_features = test_dataframe.iloc[:,:-1]
    test_features = np.array(test_features)
    test_features = pca.transform(test_features)
    results = classifier.predict(test_features)
    num_correct = (results == test_labels).sum()
    recall = num_correct *1.0/ len(test_labels)
    print "model accuracy (%): ", recall * 100, "%"
    print "Num correct", num_correct, "Total", len(test_labels)
    print classifier.score(test_features,test_labels)
    return recall * 100 
    


train_dataframe = makeTrainCSV()
train_labels = train_dataframe.label
labels = list(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = train_dataframe.iloc[:,:-1]
test_dataframe = makeTestCSV()

CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\CSaTestTWMeanSDFeatures.csv")
ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\ECTestTWMeanSDFeatures.csv")
CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\CSnTestTWMeanSDFeatures.csv")
EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\EOTestTWMeanSDFeatures.csv")
BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\BWTestTWMeanSDFeatures.csv")

result_df = pd.DataFrame(columns=['Cross_val','CSa','EC','CSn','EO','BWS','All','Description'])
for i in range(10,70,5):
    pca = getPCA(train_features,i)
    pca.fit(train_features)
    pca_train_features = pca.transform(train_features)
    for kernel in ('linear', 'poly', 'rbf'):
        print "-------------------------------------------------------------------------"
        print "kernal is: ",kernel
        classifier = svm.SVC(kernel = kernel,degree=4)
        classifier.fit(pca_train_features, train_labels)
        
        temp = []
        temp.append(calculateCrossValidateScore(kernel,pca_train_features, train_labels))
        temp.append(classfiy(CSaTest, classifier,'CSa',pca))
        temp.append(classfiy(ECTest, classifier,'EC',pca))
        temp.append(classfiy(CSnTest, classifier, 'CSn',pca))
        temp.append(classfiy(EOTest, classifier, 'EO',pca))
        temp.append(classfiy(BWSTest, classifier, 'BWS',pca))
        temp.append(classfiy(test_dataframe, classifier,'All',pca))
        temp.append(str(i) + kernel)
        result_df.loc[len(result_df)] = temp

result_df.to_csv('pca_2window_results.csv')

