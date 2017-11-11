# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 19:02:08 2017

@author: Hansika
"""
import os
import matlab.engine
import numpy as np
from audioFeatureExtraction import getFeatureForSegment
from collections import Counter

class Segmentor:
    
    def __init__(self):
        self.eng = self.startMatlab()
    
    def startMatlab(self):
        return matlab.engine.start_matlab()
    
    def getSegments(self, fileName):
        return self.eng.getSegments(fileName)
    

def classify(segments):
    results = []
    for i in segments:
        segment = np.array(i).flatten()
        features = getFeatureForSegment(segment,44100)
        results.append(classifier.predict(features)[0])
     
    birdType = Counter(results).most_common(1)[0][0]
    Probability = Counter(results).most_common(1)[0][1]*100.0/len(results)
    
    print birdType," with ", Probability, "%"


def classifyRecordings(pathToRecordings):      
    s = Segmentor()
    #segments = s.getSegments('G:\FYP Project\segmentation\Black-winged-Stilt\\Test\wav\Black-winged-Stilt Test5.wav')
    
    for root, dirs, files in os.walk(pathToRecordings):
            for file in files:
                if file.endswith(".wav"):
                    filePath = os.path.join(root, file)
                    print("processing recording " + filePath)
                    classify(s.getSegments(filePath))
            break 