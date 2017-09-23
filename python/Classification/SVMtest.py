# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 23:44:37 2017

@author: Hansika
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0,1,0,1,0,1]

clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X,y)

print(clf.predict([0.58,0.76]))

print(clf.predict([10.58,10.76]))