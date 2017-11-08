# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:43:35 2017

@author: Hansika
"""
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
        

def calculateCrossValidateScore(train_features, train_labels,clf):
    scores = cross_val_score(clf, train_features, train_labels, cv=10)
    print "mean Cross Validated Score is : ", scores.mean()
    return scores.mean()
    
    
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

def classfiywithREF(test_dataframe, feature_mask,  classifier , name):
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
    return  recall * 100 

def doRecursiveFeatureElimination(X,Y,N):
    model = RandomForestClassifierWithCoef(n_jobs=2, random_state=0)
    rfe = RFE(model, N)
    fit = rfe.fit(X, Y)
    print "Features sorted by their rank:"
    #print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
    return fit.support_


np.random.seed(0)
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
BWSTest = pd.read_csv("G:\FYP Project\GIT project\Features\\Two Window\\test\BWTestTWMeanSDFeatures.csv")
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
    feature_mask = doRecursiveFeatureElimination(train_features,train_labels,i)
    reduced_train_features = train_features[train_features.columns[feature_mask]]
    clf = RandomForestClassifierWithCoef(n_jobs=2, random_state=0)
    clf.fit(reduced_train_features, train_labels)
    temp = []
    temp.append(calculateCrossValidateScore(reduced_train_features, train_labels,clf))
    temp.append(classfiywithREF(CSaTest,feature_mask, clf,'CSa'))
    temp.append(classfiywithREF(ECTest,feature_mask, clf,'EC'))
    temp.append(classfiywithREF(CSnTest,feature_mask, clf, 'CSn'))
    temp.append(classfiywithREF(EOTest,feature_mask, clf, 'EO'))
    temp.append(classfiywithREF(BWSTest,feature_mask, clf, 'BWS'))
    temp.append(classfiywithREF(test_dataframe,feature_mask, clf,'All'))
    temp.append(i)
    result_df.loc[len(result_df)] = temp

result_df.to_csv('rf_results_2_window.csv')