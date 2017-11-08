# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 23:53:57 2017

@author: Hansika
"""
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

def calculateCrossValidateScore(kernel,train_features, train_labels):
    clf = svm.SVC(kernel = kernel,degree = 4)
    scores = cross_val_score(clf, train_features, train_labels, cv=10)
    print "mean Cross Validated Score is : ", scores.mean()
    return scores.mean()
    
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

def classfiy(test_dataframe, feature_mask,  classifier , name):
    print "for ", name 
    test_labels = test_dataframe.label
    #labels = list(set(test_labels))
    test_labels = np.array([labels.index(x) for x in test_labels])
    
    test_features = test_dataframe.iloc[:,:-1]
    test_features = test_features[test_features.columns[feature_mask]]
    test_features = np.array(test_features)

    results = classifier.predict(test_features)
    num_correct = (results == test_labels).sum()
    recall = num_correct *1.0/ len(test_labels)
    print "model accuracy (%): ", recall * 100, "%"
    print "Num correct", num_correct, "Total", len(test_labels)
    print classifier.score(test_features,test_labels)
    return recall * 100 

def doRecursiveFeatureElimination(X,Y,N):
    model = svm.SVC(kernel = 'linear')
    rfe = RFE(model, N)
    fit = rfe.fit(X, Y)
    print "Features sorted by their rank:"
    return fit.support_
    

#train_dataframe = pd.read_csv("G:\FYP Project\GIT project\Features\\train\TrainMeanSDFeatures.csv")
train_dataframe = makeTrainCSV()
train_labels = train_dataframe.label
labels = list(set(train_labels))
train_labels = np.array([labels.index(x) for x in train_labels])
train_features = train_dataframe.iloc[:,:-1]

test_dataframe = makeTestCSV()
#test_dataframe = pd.read_csv("G:\FYP Project\GIT project\Features\\test\TestMeanSDFeatures.csv")
CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\CSaTestMeanSDFeatures.csv")
ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\ECTestMeanSDFeatures.csv")
CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\CSnTestMeanSDFeatures.csv")
EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\EOTestMeanSDFeatures.csv")
BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\BWSTestMeanSDFeatures.csv")
#test_dataframe = makeTestCSV()

result_df = pd.DataFrame(columns=['Cross_val','CSa','EC','CSn','EO','BWS','All','Description'])
for i in range(10,70,5):
    print "using feaures :",i
    feature_mask = doRecursiveFeatureElimination(train_features,train_labels,i)
    reduced_train_features = train_features[train_features.columns[feature_mask]]
    for kernel in ('linear', 'poly', 'rbf'):
        print "-------------------------------------------------------------------------"
        print "kernal is: ",kernel
        classifier = svm.SVC(kernel = kernel,degree=4)
        classifier.fit(reduced_train_features, train_labels)
        temp = []
        temp.append(calculateCrossValidateScore(kernel,reduced_train_features, train_labels))
        temp.append(classfiy(CSaTest,feature_mask, classifier,'CSa'))
        temp.append(classfiy(ECTest,feature_mask, classifier,'EC'))
        temp.append(classfiy(CSnTest,feature_mask, classifier, 'CSn'))
        temp.append(classfiy(EOTest,feature_mask, classifier, 'EO'))
        temp.append(classfiy(BWSTest,feature_mask, classifier, 'BWS'))
        temp.append(classfiy(test_dataframe,feature_mask, classifier,'All'))
        temp.append(str(i) + kernel)
        result_df.loc[len(result_df)] = temp

result_df.to_csv('rfe_1window_results.csv')

