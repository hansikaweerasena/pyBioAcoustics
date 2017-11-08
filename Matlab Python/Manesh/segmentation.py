# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 00:02:56 2017

@author: Manesh
"""

import matlab.engine
import numpy as np
from audioFeatureExtraction import getFeatureForSegment
from collections import Counter
import pickle

class Segmentor:
    
    def __init__(self):
        self.eng = self.startMatlab()
    
    def startMatlab(self):
        return matlab.engine.start_matlab()
    
    def getSegments(self, fileName):
        return self.eng.getSegments(fileName)

        
s = Segmentor()
segments = s.getSegments('G:\FYP Project\segmentation\Black-winged-Stilt\\Test\wav\Black-winged-Stilt Test5.wav')

classier = pickle.load(open('techno_model.sav', 'rb'))

results = []
for i in segments:
    segment = np.array(i).flatten()
    features = getFeatureForSegment(segment,44100)
    results.append(classifier.predict(features)[0])
 
birdType = Counter(results).most_common(1)[0][0]
Probability = Counter(results).most_common(1)[0][1]*100.0/len(results)

print birdType," with ", Probability, "%"