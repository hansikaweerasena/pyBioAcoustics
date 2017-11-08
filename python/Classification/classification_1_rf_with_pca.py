# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 12:35:09 2017

@author: Hansika
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
        

def calculateCrossValidateScore(train_features, train_labels,clf):
    scores = cross_val_score(clf, train_features, train_labels, cv=10)
    print "mean Cross Validated Score is : ", scores.mean()
    return scores.mean()
    
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

def makeTestCSV():
    CSaTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\CSaTestMeanSDFeatures.csv")
    ECTest =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\ECTestMeanSDFeatures.csv")
    CSnTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\CSnTestMeanSDFeatures.csv")
    EOTest =  pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\EOTestMeanSDFeatures.csv")
    BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\One Window\\test\BWSTestMeanSDFeatures.csv")
    
    return pd.concat([CSaTest, ECTest,CSnTest,EOTest,BWSTest])

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
    return  recall * 100 

def classfiywithREF(test_dataframe, pca,  classifier , name):
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
    return  recall * 100 


np.random.seed(0)
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



classifier = RandomForestClassifier(n_jobs=2, random_state=0)
classifier.fit(train_features, train_labels)

result_df = pd.DataFrame(columns=['Cross_val','CSa','EC','CSn','EO','BWS','All','Description'])
temp = []
temp.append(calculateCrossValidateScore(train_features, train_labels,classifier))
temp.append(classfiy(CSaTest, classifier,'CSa'))
temp.append(classfiy(ECTest, classifier,'EC'))
temp.append(classfiy(CSnTest, classifier, 'CSn'))
temp.append(classfiy(EOTest, classifier, 'EO'))
temp.append(classfiy(BWSTest, classifier, 'BWS'))
temp.append(classfiy(test_dataframe, classifier,'All'))
temp.append('original')
result_df.loc[len(result_df)] = temp

for i in range(10,70,5):
    print "using feaures :",i
    pca = getPCA(train_features,i)
    pca.fit(train_features)
    pca_train_features = pca.transform(train_features)
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(pca_train_features, train_labels)
    temp = []
    temp.append(calculateCrossValidateScore(pca_train_features, train_labels,clf))
    temp.append(classfiywithREF(CSaTest,pca, clf,'CSa'))
    temp.append(classfiywithREF(ECTest,pca, clf,'EC'))
    temp.append(classfiywithREF(CSnTest,pca, clf, 'CSn'))
    temp.append(classfiywithREF(EOTest,pca, clf, 'EO'))
    temp.append(classfiywithREF(BWSTest,pca, clf, 'BWS'))
    temp.append(classfiywithREF(test_dataframe,pca, clf,'All'))
    temp.append(i)
    result_df.loc[len(result_df)] = temp

result_df.to_csv('rf_with_pca_results_1_window.csv')